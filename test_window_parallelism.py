# TODO: get this import to work inside test/
from distributed_utils import (
    print_in_order, print_rank0, get_backend, get_device_type
)
from window_parallelism import window_parallel_shift
from ulysses import _SeqAllToAll

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from einops import rearrange
from functools import partial



## TODO: follow the best practice and make the unit-test more modular
def setup_window_parallelism(
    ulysses_degree=1, 
    window_parallelism_degree=1, 
):
    r'''initialize toy data for window parallelism.
    
        fully_sharded_input: 
            (B, n_total_patches/wp/sp, hc, hs) if SP > 1 else: (B, n_row, n_col, hc, hs)
    '''
    ## Initialize 3D device mesh
    world_size = dist.get_world_size()
    SP = ulysses_degree
    WP = window_parallelism_degree
    assert world_size % SP % WP == 0, 'Data Parallism group is uneven'
    DP = world_size // SP // WP
    mesh_shape = [SP, WP, DP]
    device = get_device_type()
    device_mesh = init_device_mesh(device, mesh_shape, mesh_dim_names=['SP', 'WP', 'DP'])

    ## Get ProcessGroup attributes
    sp_group = device_mesh.get_group('SP')
    wp_group = device_mesh.get_group('WP')
    dp_group = device_mesh.get_group('DP')
    comm_groups = (dp_group, wp_group, sp_group, )
    wp_rank = dist.get_rank(wp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_rank = dist.get_rank(sp_group)
    global_world_size = dist.get_world_size()
    ## TODO: cleaner way to print all the ranks? 
    print_in_order(f"WP RANK: {wp_rank}")
    print_in_order(f"SP RANK: {sp_rank}")
    print_in_order(f"DP RANK: {dp_rank}")

    ## Set image dimensions 
    B, hc, hs = (2, SP, 2)
    n_row, n_col = (global_world_size, global_world_size)  # grid size of windows
    win_h, win_w = window_size = (2, 2)  # window size
    assert (win_h % 2 == 0) and (win_w % 2 == 0), \
        'window size needs to be even numbers in order to shift' \
        'twice to return to the original grid'

    ## Derive other image related dimensions
    n_row_patches, n_col_patches = patch_grid_size = n_row*win_h, n_col*win_w
    input_dim = (B, n_row_patches, n_col_patches, hc, hs)
    print_rank0(f"window_size: {window_size}")
    print_rank0(f"num_row_patches: {n_row_patches}")
    print_rank0(f"num_col_patches: {n_col_patches}")
    print_rank0(f'Image dim can be ({n_row_patches}, {n_col_patches})'
                ' * patch_dim\n\n')

    ## Create toy image -> (B, n_row_patches, n_col_patches, hc, hs)
    total_dim = 1
    for dim in input_dim:
        total_dim *= dim
    # TODO: set device, dtype?
    dtype = torch.get_default_dtype()
    input = torch.arange(total_dim, dtype=dtype).view(input_dim)  

    ## Shard input by row -> (B, n_row_patches/wp, n_col_patches, hc, hs)
    assert n_row % WP == 0, \
        'number of windows across row needs to be divisible by window ' \
        'parallel degree'
    ## TODO: Can make it more clear in the code why n_row needs to be divisible.
    row_chunk_size = n_row_patches // WP
    row_partition_slice = slice(row_chunk_size*wp_rank, row_chunk_size*(wp_rank+1))
    row_partitioned_input = input[:, row_partition_slice, :, :, :]

    ## further shard input for sequence parallelism (ulysses) 
    if SP > 1:  # -> (B, n_total_patches/wp/sp, hc, hs)
        flattened_input = row_partitioned_input.flatten(1, 2)
        B, s, hc, hs = flattened_input.shape
        assert s % SP == 0, \
            'uneven sequence length ulysses is not yet supported'
        assert hc % SP == 0, 'uneven head count ulysses is not yet supported'
        seq_split_input = flattened_input.tensor_split(SP, dim=1)
        fully_sharded_input = seq_split_input[sp_rank]
    else:
        fully_sharded_input = row_partitioned_input

    return input, fully_sharded_input, patch_grid_size, window_size, comm_groups

def test_intranode_window_parallel_shift():
    r'''tests window_parallel_shift
    input[0, :, :, 0, 0]:
    tensor([[  0,   4,   8,  12,  16,  20,  24,  28],
            [ 32,  36,  40,  44,  48,  52,  56,  60],
            [ 64,  68,  72,  76,  80,  84,  88,  92],
            [ 96, 100, 104, 108, 112, 116, 120, 124],
            [128, 132, 136, 140, 144, 148, 152, 156],
            [160, 164, 168, 172, 176, 180, 184, 188],
            [192, 196, 200, 204, 208, 212, 216, 220],
            [224, 228, 232, 236, 240, 244, 248, 252]])
    shifted_input[0, :, :, 0, 0]:
    tensor([[ 36,  40,  44,  48,  52,  56,  60,  32],
            [ 68,  72,  76,  80,  84,  88,  92,  64],
            [100, 104, 108, 112, 116, 120, 124,  96],
            [132, 136, 140, 144, 148, 152, 156, 128],
            [164, 168, 172, 176, 180, 184, 188, 160],
            [196, 200, 204, 208, 212, 216, 220, 192],
            [228, 232, 236, 240, 244, 248, 252, 224],
            [  4,   8,  12,  16,  20,  24,  28,   0]])
    gathered_shifted_input[0, :, :, 0, 0]:
    tensor([[ 36,  40,  44,  48,  52,  56,  60,  32],
            [ 68,  72,  76,  80,  84,  88,  92,  64],
            [100, 104, 108, 112, 116, 120, 124,  96],
            [132, 136, 140, 144, 148, 152, 156, 128],
            [164, 168, 172, 176, 180, 184, 188, 160],
            [196, 200, 204, 208, 212, 216, 220, 192],
            [228, 232, 236, 240, 244, 248, 252, 224],
            [  4,   8,  12,  16,  20,  24,  28,   0]])
    '''
    ## Init distributed backend
    dist.init_process_group(backend=get_backend())
    world_size = dist.get_world_size()
    # input: (B, n_row_patches, n_col_patches, hc, hs)
    # row_partitioned_input: (B, n_row_patches/wp, n_col_patches, hc, hs)
    input, row_partitioned_input, patch_grid_size, window_size, comm_groups = \
        setup_window_parallelism(window_parallelism_degree=world_size)
    dp_group, wp_group, sp_group = comm_groups
    win_h, win_w = window_size
    shift_size = win_h//2, win_w//2
    WP = dist.get_world_size(wp_group)
    wp_rank = dist.get_rank(wp_group)
    wp_world_size = dist.get_world_size(wp_group)

    ## Initialize gradients in-place
    input.requires_grad_(True)
    row_partitioned_input.requires_grad_(True)
    
    ## Normally shift image by shift_size -> ()
    shifted_input = torch.roll(input, (-shift_size[0], -shift_size[1]), dims=(1, 2))
    print_rank0(f"input[0, :, :, 0, 0]:\n{input[0, :, :, 0, 0].int()}\n")
    
    ## shift windows across GPUs using window_parallel_shift 
    # -> (B, n_row_patches/wp, n_col_patches, hc, hs)
    shifted_row_input = window_parallel_shift(row_partitioned_input, (-shift_size[0], -shift_size[1]), wp_group)

    ## verify the shift with all-gather -> (B, n_row_patches/wp, n_col_patches, hc, hs)
    nrow_dim = 1
    out_lst = [torch.empty_like(shifted_row_input) for _ in range(WP)]
    dist.all_gather(out_lst, shifted_row_input, group=wp_group)
    gathered_shifted_input = torch.cat(out_lst, dim=nrow_dim)
    print_rank0(f'shifted_input[0, :, :, 0, 0]:\n'
                f'{shifted_input[0, :, :, 0, 0].int()}\n')
    print_rank0(f'gathered_shifted_input[0, :, :, 0, 0]:\n'
                f'{gathered_shifted_input[0, :, :, 0, 0].int()}\n')
    assert torch.allclose(gathered_shifted_input, shifted_input), \
        'unit-test for shift mechanism failed'

    ## TODO: Do attention per window -> (B, hc, s, hs)
    reshape_for_FA = partial(
        rearrange, 
        pattern='B (n_row w1) (n_col w2) hc hs -> B (n_row n_col) hc (w1 w2) hs',
        w1 = win_h,
        w2 = win_w
    )
    reshaped_row_input = reshape_for_FA(shifted_row_input)  
    reshaped_input = reshape_for_FA(shifted_input)  # (B, hc, s, hs)
    wp_out = F.scaled_dot_product_attention(*[reshaped_row_input for _ in range(3)])  ## FIXME: Make sure that having the extra dim N_w in (B, N_w, hc, S, hs) works as intended for FA.
    out = F.scaled_dot_product_attention(*[reshaped_input for _ in range(3)])
    wp_out_lst = [torch.empty_like(wp_out) for _ in range(WP)]
    dist.all_gather(wp_out_lst, wp_out, group=wp_group)
    num_win_dim = 1
    full_wp_out = torch.cat(wp_out_lst, dim=num_win_dim)
    assert torch.allclose(full_wp_out, out),\
        'FA forward of window parallelism is incorrect'
    
    print_rank0('🎉 part (1/2) all test for intra-node window parallelism (fwd) passed 🎉')

    label = torch.randn_like(out)
    wp_label = label.chunk(wp_world_size, dim=num_win_dim)[wp_rank]
    loss = F.mse_loss(out, label)
    # Divide by wp_world_size in order for the gradient to match up. This is because the gradient will has a difference in constant dividor N. 
    wp_loss = F.mse_loss(wp_out, wp_label) / wp_world_size
    loss.backward()
    wp_loss.backward()
    
    print_rank0(f"row_partitioned_input.grad[0, :, :, 0, 0]: \n{row_partitioned_input.grad[0, :, :, 0, 0]}")
    print_rank0(f"input.grad[0, :, :, 0, 0]: \n{input.grad[0, :, :, 0, 0]}")
    
    wp_input_grad_lst = [torch.empty_like(row_partitioned_input) for _ in range(WP)]
    dist.all_gather(wp_input_grad_lst, row_partitioned_input.grad, group=wp_group)
    wp_input_grad = torch.cat(wp_input_grad_lst, dim=nrow_dim)
    assert torch.allclose(wp_input_grad, input.grad), \
        'input gradient and wp_input gradient doesnt match up'

    print_rank0('🎉 part (2/2) all test for intra-node window parallelism (bwd) passed 🎉')


def test_WP_shift_and_rev_shift():
    r'''tests window_parallel_shift
    input[0, :, :, 0, 0]:
    tensor([[  0,   4,   8,  12,  16,  20,  24,  28],
            [ 32,  36,  40,  44,  48,  52,  56,  60],
            [ 64,  68,  72,  76,  80,  84,  88,  92],
            [ 96, 100, 104, 108, 112, 116, 120, 124],
            [128, 132, 136, 140, 144, 148, 152, 156],
            [160, 164, 168, 172, 176, 180, 184, 188],
            [192, 196, 200, 204, 208, 212, 216, 220],
            [224, 228, 232, 236, 240, 244, 248, 252]])
    shifted_input[0, :, :, 0, 0]:
    tensor([[ 36,  40,  44,  48,  52,  56,  60,  32],
            [ 68,  72,  76,  80,  84,  88,  92,  64],
            [100, 104, 108, 112, 116, 120, 124,  96],
            [132, 136, 140, 144, 148, 152, 156, 128],
            [164, 168, 172, 176, 180, 184, 188, 160],
            [196, 200, 204, 208, 212, 216, 220, 192],
            [228, 232, 236, 240, 244, 248, 252, 224],
            [  4,   8,  12,  16,  20,  24,  28,   0]])
    gathered_shifted_input[0, :, :, 0, 0]:
    tensor([[ 36,  40,  44,  48,  52,  56,  60,  32],
            [ 68,  72,  76,  80,  84,  88,  92,  64],
            [100, 104, 108, 112, 116, 120, 124,  96],
            [132, 136, 140, 144, 148, 152, 156, 128],
            [164, 168, 172, 176, 180, 184, 188, 160],
            [196, 200, 204, 208, 212, 216, 220, 192],
            [228, 232, 236, 240, 244, 248, 252, 224],
            [  4,   8,  12,  16,  20,  24,  28,   0]])
    '''
    ## Init distributed backend
    dist.init_process_group(backend=get_backend())
    world_size = dist.get_world_size()
    # input: (B, n_row_patches, n_col_patches, hc, hs)
    # row_partitioned_input: (B, n_row_patches/wp, n_col_patches, hc, hs)
    input, row_partitioned_input, patch_grid_size, window_size, comm_groups = \
        setup_window_parallelism(window_parallelism_degree=world_size)
    input, row_partitioned_input = input.detach(), row_partitioned_input.detach()
    sp_group, wp_group, dp_group = comm_groups
    win_h, win_w = window_size
    shift_size = win_h//2, win_w//2
    WP = dist.get_world_size(wp_group)
    
    ## shift windows across GPUs using window_parallel_shift 
    # -> (B, n_row_patches/wp, n_col_patches, hc, hs)
    shifted_row_input = window_parallel_shift(row_partitioned_input, shift_size, wp_group)
    double_shifted_row_input = window_parallel_shift(shifted_row_input, (-shift_size[0], -shift_size[1]), wp_group)
    shifted_input = torch.empty_like(input[0])
    dist.all_gather_into_tensor(shifted_input, shifted_row_input[0])
    double_shifted_input = torch.empty_like(input[0])
    dist.all_gather_into_tensor(double_shifted_input, double_shifted_row_input[0])

    print_rank0('🎉 unit-test for shift and then rev-shift passed 🎉')
    

def test_internode_window_parallel_shift():
    ## Init distributed backend
    dist.init_process_group(backend=get_backend())
    assert dist.get_world_size() == 24, \
        'assuming world_size == 24'
    # input: (B, n_row_patches, n_col_patches, hc, hs)
    # seq_sharded_x: (B, s/sp, hc, hs) with s = n_row_patches * n_col_patches / wp
    input, seq_sharded_x, patch_grid_size, shift_size, comm_groups = \
        setup_window_parallelism(ulysses_degree=12, window_parallelism_degree=2)
    dp_group, wp_group, sp_group = comm_groups
    B, loc_s, hc, hs = seq_sharded_x.shape
    WP = dist.get_world_size(wp_group)
    SP = dist.get_world_size(sp_group)
    # wp_world_size = dist.get_world_size(wp_group)

    ## Normally shift window:
    shifted_input = torch.roll(input, (-shift_size[0], -shift_size[1]), dims=(1, 2))

    ## gather seq and scatter hc using all2all from Ulysses
    gather_idx = 1
    scatter_idx = 2
    batch_dim_idx = 0
    # -> (B, s, hc/sp, hs)
    seq_gathered_x = _SeqAllToAll.apply(sp_group, seq_sharded_x, scatter_idx, 
                                        gather_idx, batch_dim_idx)
    # -> (B, n_row_patches/WP, n_col_patches, hc/SP, hs)
    unflattened_sharded_input = seq_gathered_x.view(B, patch_grid_size[0]//WP, 
                                                    patch_grid_size[1], hc//SP, hs)
    # -> (B, n_row_patches/WP, n_col_patches, hc/SP, hs)
    shifted_shards = window_parallel_shift(unflattened_sharded_input, shift_size, 
                                               wp_group)

    # TODO: do attention

    ## Gather across WP and SP to compare shift results
    # B, loc_r, c, loc_hc, hs = shifited_row_input.shape
    # gather along row (WP)
    row_win_dim = 1
    hc_dim = 3
    row_gathered_shifted_lst = [torch.empty_like(shifted_shards) for _ in range(WP)]
    dist.all_gather(row_gathered_shifted_lst, shifted_shards, group=wp_group)
    shifted_hc_shards = torch.cat(row_gathered_shifted_lst, dim=row_win_dim)
    # gather along hc (SP)
    hc_gathered_shifted_lst = [torch.empty_like(shifted_hc_shards) for _ in range(SP)]
    dist.all_gather(hc_gathered_shifted_lst, shifted_hc_shards, group=sp_group)
    fully_gathered_shifted_input = torch.cat(hc_gathered_shifted_lst, dim=hc_dim)

    assert torch.allclose(shifted_input, fully_gathered_shifted_input)
    print_rank0('🎉 all test for inter-node window parallelism passed 🎉')

# def test_ulysses():
#     ## Init distributed backend
#     dist.init_process_group(backend=get_backend())
#     sp_rank = dist.get_rank()
#     sp_world_size = dist.get_world_size()
#     print_in_order(f"RANK: {sp_rank}")
#     print_rank0(f"WORLD_SIZE: {sp_world_size}\n\n")

#     # s, B, h = [sp_world_size, 4, sp_world_size]
#     B, s, h = [4, sp_world_size, sp_world_size]
#     h = 3*h # qkv
#     input = torch.randn(B, s, h)
#     gather_idx = 1
#     scatter_idx = 2
    
#     ## TODO: customize seqall2all for better readability
#     copy_input = input.clone()
#     first_all2all = _SeqAllToAll.apply(input, scatter_idx, gather_idx)  # -> (B, s, hc)
#     print(f"first_all2all.shape: {first_all2all.shape}")

    ## Attention
    # second_all2all = _SeqAllToAll.apply(first_all2all, gather_idx, scatter_idx)

    # print(f"second_all2all.shape: {second_all2all.shape}")
    # print(f"copy_input.shape: {copy_input.shape}")

if __name__ == "__main__":
    torch.manual_seed(42999)
    # torch.set_default_dtype(torch.float32)
    test_intranode_window_parallel_shift()
    # test_WP_shift_and_rev_shift()
    # test_internode_window_parallel_shift()
    # test_ulysses()
