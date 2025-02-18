import torch
import torch.distributed as dist
from einops import rearrange
# TODO: how to get this import to work inside test/
from distributed_utils import (
    print_in_order, print_rank0, get_backend
)
from model_parallelism import window_parallel_shift


def test_window_parallel_shift():
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
    wp_rank = dist.get_rank()
    wp_world_size = dist.get_world_size()
    print_in_order(f"RANK: {wp_rank}")
    print_rank0(f"WORLD_SIZE: {wp_world_size}\n\n")

    ## Set image dimensions 
    B, hc, hs = (4, 2, 2)
    n_row, n_col = (wp_world_size, wp_world_size)  # Window Grid Dim
    win_h, win_w = (2, 2)  # Single Window Dim
    assert (win_h % 2 == 0) and (win_w % 2 == 0), \
        'window size needs to be even numbers in order to shift' \
        'twice to return to the original grid'

    ## Derive other dimensions
    shift_size = win_h//2, win_w//2
    num_patches_row, num_patches_col = n_row*win_h, n_col*win_w
    input_dim = (B, num_patches_row, num_patches_col, hc, hs)
    print_rank0(f"window_size: {win_h, win_w}")
    print_rank0(f"shift_size: {shift_size}")
    print_rank0(f"num_row_patches: {num_patches_row}")
    print_rank0(f"num_col_patches: {num_patches_col}")
    print_rank0(f'Image dim can be ({num_patches_row}, {num_patches_col})'
                ' * patch_dim\n\n')

    ## Create toy image -> (B, n_row_patches, n_col_patches, hc, hs)
    total_dim = 1
    for dim in input_dim:
        total_dim *= dim
    input = torch.arange(total_dim).view(input_dim)  # TODO: set device, dtype?

    ## Normally shift image by shift_size
    rev_shift_size = -shift_size[0], -shift_size[1]
    shifted_input = torch.roll(input, rev_shift_size, dims=(1, 2))
    print_rank0(f"input[0, :, :, 0, 0]:\n{input[0, :, :, 0, 0]}\n")

    ## Shard input by row -> (B, n_row_patches/wp, n_col_patches, hc, hs)
    assert n_row % wp_world_size == 0, \
        'number of windows across row needs to be divisible by window ' \
        'parallel degree'
    ## TODO: Can make it more clear in the code why n_row needs to be divisible.
    row_chunk_size = num_patches_row // wp_world_size
    row_partition_slice = slice(row_chunk_size*wp_rank, row_chunk_size*(wp_rank+1))
    row_partitioned_input = input[:, row_partition_slice, :, :, :]
    
    ## shift windows across GPUs using window_parallel_shift 
    shifted_row_input = window_parallel_shift(row_partitioned_input, shift_size)

    ## TODO: Do attention per window

    ## all-gather to verify the shift
    out_lst = [torch.empty_like(shifted_row_input) for _ in range(wp_world_size)]
    dist.all_gather(out_lst, shifted_row_input)
    gathered_shifted_input = torch.cat(out_lst, dim=1)

    print_rank0(f'shifted_input[0, :, :, 0, 0]:\n'
                f'{shifted_input[0, :, :, 0, 0]}\n')
    print_rank0(f'gathered_shifted_input[0, :, :, 0, 0]:\n'
                f'{gathered_shifted_input[0, :, :, 0, 0]}\n')
    assert torch.allclose(gathered_shifted_input, shifted_input)
    print_rank0('🎉 distributed shift window test passed 🎉')

if __name__ == "__main__":
    test_window_parallel_shift()