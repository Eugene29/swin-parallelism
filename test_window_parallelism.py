# TODO: get this import to work inside test/
from mpi4py import MPI
from distributed_utils import *  # TODO: Change it to explicit imports
from window_parallelism import window_parallel_shift
from ulysses import _SeqAllToAll

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
if torch.xpu.is_available():
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch
from einops import rearrange
from functools import partial

import os
import numpy as np
import logging
from collections import namedtuple


def log_tensor_slice(tensor_name, tensor):
    """Log a slice of tensor on rank 0

    Args:
        tensor: [B, nrow_patch, ncol_patch, hc, hs]
    """
    if dist.get_rank() == 0:
        logging.info(f"{tensor_name}:\n{tensor[0, :, :, 0, 0].long()}\n\n")


def setup(ulysses_degree=1, window_parallelism_degree=1):
    r"""Initialize comm groups. Create and return both row sharded and head count sharded 
    toy data.
    
    Returns:
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    """

    ## Initialize 3D Device Mesh
    if 'RANK' in os.environ:  # Torchrun
        dist.init_process_group(backend=get_backend())  
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:  # Mpiexec
        rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
        dist.init_process_group(
            backend=get_backend(), rank=rank, world_size=world_size
        )
    SP, WP = (ulysses_degree, window_parallelism_degree)
    DP = world_size // SP // WP
    assert world_size == (DP*SP*WP), 'Your parallel degrees are likely wrong'
    device_mesh = init_device_mesh(device_type=get_device_type(), 
                                   mesh_shape=[DP, WP, SP],
                                   mesh_dim_names=['DP','WP','SP'])
    wp_group = device_mesh.get_group('WP')
    sp_group = device_mesh.get_group('SP')
    dp_group = device_mesh.get_group('DP')
    wp_rank = dist.get_rank(wp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_rank = dist.get_rank(dp_group)
    dist.all_reduce(torch.randn(100), group=wp_group)  # Init send/recv comm group

    ## Set Image Dimensions
    B, hc, hs = (DP, SP, 2)
    nrow, ncol = (2*WP, 2*WP)  # grid size of windows
    win_h, win_w = (2, 3)  # window size
    shift_h, shift_w = (2, 1)
    nrow_patch, ncol_patch = nrow*win_h, ncol*win_w
    input_dim = (B, nrow_patch, ncol_patch, hc, hs)
    args = namedtuple(
        'args', ['win_h', 'win_w', 'shift_h', 'shift_w'],
        defaults=[win_h, win_w, shift_h, shift_w]
    )()

    ## Create Toy Image -> B, n_patch_row, n_patch_col, hc, hs
    torch.set_default_device(rank % get_device_count())
    dtype = torch.get_default_dtype()
    input = torch.arange(np.prod(input_dim), dtype=dtype).view(input_dim)  

    ## WP Sharding (shard by row) -> B, n_patch_row/wp, n_patch_col, hc, hs
    assert nrow % WP == 0, 'num row windows must be divisible by WP'
    row_chunk = nrow // WP * win_h  # num row patches per WP
    row_slice = slice(row_chunk*wp_rank, row_chunk*(wp_rank+1))
    input_row_shard = input[:, row_slice, :, :, :]

    ## SP Sharding (shard by head count) -> B, n_patch_row/wp, n_patch_col, hc/sp, hs
    assert (row_chunk*ncol_patch) % SP == 0, 'uneven sequence ulysses is not supported'
    assert hc % SP == 0, 'uneven head ulysses is not supported'
    hc_chunk = hc // SP  # num head count per SP(ulysses)
    hc_slice = slice(hc_chunk*sp_rank, hc_chunk*(sp_rank+1))
    input_hc_shard = input_row_shard[:, :, :, hc_slice, :]

    ## DP Sharding (shard by batch) -> B/DP, n_patch_row/wp, n_patch_col, hc/sp, hs
    assert B % DP == 0, 'Batch size indivisible by DP'
    dp_chunk = B // DP  # num head count per SP(ulysses)
    dp_slice = slice(dp_chunk*dp_rank, dp_chunk*(dp_rank+1))
    input_shard = input_hc_shard[dp_slice, :, :, :, :]

    log_tensor_slice('input', input)
    log_tensor_slice('input_shard', input_row_shard)

    # test = gather_3D_shard(input_shard.contiguous(), device_mesh)
    # raise KeyboardInterrupt()
    return input, input_shard, device_mesh, args


def gather_3D_shard(input_shard, device_mesh, dp_dim=0, wp_dim=1, sp_dim=3):
    wp_group = device_mesh['WP'].get_group()
    sp_group = device_mesh['SP'].get_group()
    dp_group = device_mesh['DP'].get_group()
    WP = dist.get_world_size(wp_group)
    SP = dist.get_world_size(sp_group)
    DP = dist.get_world_size(dp_group)
    input_shard = input_shard.contiguous()

    # gather along WP
    lst_wp_shard = [torch.empty_like(input_shard) for _ in range(WP)]
    dist.all_gather(lst_wp_shard, input_shard, group=wp_group)
    hc_shard = torch.cat(lst_wp_shard, dim=wp_dim)  # concat row dim
    # gather along SP
    lst_hc_shard = [torch.empty_like(hc_shard) for _ in range(SP)]
    dist.all_gather(lst_hc_shard, hc_shard, group=sp_group)
    dp_shard = torch.cat(lst_hc_shard, dim=sp_dim)  # concat hc dim
    # gather along DP
    lst_dp_shard = [torch.empty_like(dp_shard) for _ in range(DP)]
    dist.all_gather(lst_dp_shard, dp_shard, group=dp_group)
    gathered_shifted = torch.cat(lst_dp_shard, dim=dp_dim)  # concat batch dim

    return gathered_shifted
    

def test_window_parallel_shift(input, input_shard, device_mesh, args):
    """Test window parallel shift against normal shift (roll)
    Args:
        input: B, n_patch_row, n_patch_col, hc, hs
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    """

    neg_shift_size = (-args.shift_h, -args.shift_w)
    wp_group = device_mesh['WP'].get_group()

    # Normal shift (roll)
    shifted = torch.roll(input, neg_shift_size, dims=(1,2))  # dim: nrow, ncol

    # Window Parallel Shift
    wp_shifted = window_parallel_shift(input_shard, neg_shift_size, wp_group=wp_group)
    # raise KeyboardInterrupt()
    ## Compare (by gathering parallel input)
    # DP_dim=0, WP_dim=1, HC_dim=3
    # raise KeyboardInterrupt()
    gathered_shifted = gather_3D_shard(wp_shifted, device_mesh, dp_dim=0, wp_dim=1, sp_dim=3)
    log_tensor_slice('shifted', shifted)
    log_tensor_slice('gathered_shifted', gathered_shifted)
    assert torch.allclose(shifted, gathered_shifted)
    print_rank0("passed test_window_parallel_shift")


def test_window_parallel_shift_and_rev_shift(input, input_shard, device_mesh, args):
    """Test whether shift and rev_shift returns to the same tensor
    Args:
        input: B, n_patch_row, n_patch_col, hc, hs
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    """

    shift_h, shift_w = (args.shift_h, args.shift_w)
    wp_group = device_mesh['WP'].get_group()

    ## Shift and Reverse Shift
    shifted1 = window_parallel_shift(input_shard, (-shift_h, -shift_w), wp_group=wp_group)
    shifted2 = window_parallel_shift(shifted1, (shift_h, shift_w), wp_group=wp_group)

    assert torch.allclose(input_shard, shifted2)
    print_rank0("passed test_window_parallel_shift_and_rev_shift")


def test_ulysses(input, input_shard, device_mesh, args):
    """Test whether two all2all leads to original input

    Note: 
        Here, the order of two all2alls are flipped, but its good enough for a 
        testing purposes
    Args:
        input: B, n_patch_row, n_patch_col, hc, hs
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    """

    sp_group = device_mesh['SP'].get_group()
    flattened_shard = input_shard.flatten(1, 2)  # -> B, s, hc/sp, hs
    scatter_idx, gather_idx = 1, 3

    all2all1 = _SeqAllToAll.apply(sp_group, flattened_shard, scatter_idx, gather_idx, 0)
    all2all2 = _SeqAllToAll.apply(sp_group, all2all1, gather_idx, scatter_idx, 0)

    assert torch.allclose(flattened_shard, all2all2)
    print_rank0("passed test_ulysses")


def test_WP_bwd(input, input_shard, device_mesh, args):
    neg_shift_size = (-args.shift_h, -args.shift_w)
    dp_group = device_mesh['DP'].get_group()
    wp_group = device_mesh['WP'].get_group()
    sp_group = device_mesh['SP'].get_group()
    dp_rank = dist.get_rank(dp_group)
    wp_rank = dist.get_rank(wp_group)
    sp_rank = dist.get_rank(sp_group)
    input.requires_grad_()  # in place operation to set requires_grad = True
    input_shard.requires_grad_()
    dp_B, wp_nrow, ncol, sp_hc, hs = input_shard.shape

    # Normal shift (roll)
    shifted = torch.roll(input, neg_shift_size, dims=(1,2))  # dim: nrow, ncol

    # Window Parallel Shift
    wp_shifted = window_parallel_shift(input_shard, neg_shift_size, wp_group=wp_group)

    ## Random Compute
    matrix = torch.randn_like(shifted).transpose(-1, -2)
    slice_row = slice(wp_nrow*wp_rank, wp_nrow*(wp_rank+1))
    slice_hc = slice(sp_hc*sp_rank, sp_hc*(sp_rank+1))
    slice_dp = slice(dp_B*dp_rank, dp_B*(dp_rank+1))
    wp_matrix = matrix[slice_dp, slice_row, :, slice_hc, :]
    out = shifted * matrix
    wp_out = wp_shifted * wp_matrix
    gathered_out = gather_3D_shard(wp_out, device_mesh) 
    assert torch.allclose(out, gathered_out)

    ## Loss
    label = torch.randn_like(out)
    wp_label = label[slice_dp, slice_row, :, slice_hc, :]
    loss = F.mse_loss(label, out)
    wp_loss = F.mse_loss(wp_label, wp_out)
    wp_loss_tmp = wp_loss.detach().clone()
    dist.all_reduce(wp_loss_tmp) 
    wp_loss_avg = wp_loss_tmp / dist.get_world_size()
    assert torch.allclose(loss, wp_loss_avg)

    ## BWD
    loss.backward()
    wp_loss.backward()
    gathered_grad = gather_3D_shard(input_shard.grad, device_mesh)
    log_tensor_slice('input.grad', input.grad)
    log_tensor_slice('gathered_grad', gathered_grad)
    assert torch.allclose(input.grad, gathered_grad)

    ## Compare (by gathering parallel input)
    # DP_dim=0, WP_dim=1, HC_dim=3
    # gathered_shifted = gather_3D_shard(wp_shifted, device_mesh, 0, 1, 3)

# def test_WP_fwd_bwd(input, input_shard, device_mesh, args):
#     from window_parallelism import _WindowParallelism
#     pos_shift_size = (args.shift_h, args.shift_w)
#     neg_shift_size = (-args.shift_h, -args.shift_w)
#     wp_group = device_mesh['WP'].get_group()
#     ctx = namedtuple('ctx', ['shift_size', 'wp_group', 'wp_group'])()
#     fwd_out = _WindowParallelism().forward(ctx, input_shard, neg_shift_size, wp_group)
#     bwd_out = _WindowParallelism().backward(ctx, fwd_out, pos_shift_size, wp_group)

    # assert torch.allclose(fwd_out, bwd_out)

# def test_3D_fwd_bwd(input, input_shard, device_mesh, args):
#     """Roughly simulate the workload of SWIN transformer and check the functionality of 
#     fwd+bwd
    
#     Args:
#         input: B, n_patch_row, n_patch_col, hc, hs
#         input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
#     """

#     shift_h, shift_w = args.shift_h, args.shift_w
#     w_h, w_w = args.win_h, args.win_w
#     input.requires_grad_()  # in place operation to set requires_grad = True
#     input_shard.requires_grad_()
#     wp_group = device_mesh['WP'].get_group()
#     sp_group = device_mesh['SP'].get_group()
#     dp_group = device_mesh['DP'].get_group()
#     wp_rank = dist.get_rank(wp_group)
#     sp_rank = dist.get_rank(sp_group)
#     dp_rank = dist.get_rank(dp_group)
#     rearrange_for_fa = partial(rearrange,
#         pattern='B (n_win_r w1) (n_win_c w2) hc hs -> B (n_win_r n_win_c) hc (w1 w2) hs',
#         w1=w_h,
#         w2=w_w,
#     )

#     ## Normal FWD
#     shifted = torch.roll(input, (-shift_h, -shift_w), dims=(1,2))  # dims: nrow, ncol
#     rearranged = rearrange_for_fa(shifted)
#     q, k, v = [rearranged*(c+1) for c in range(3)]  # arbitrary q, k, v
#     out = F.scaled_dot_product_attention(q, k, v)

#     ## 3D FWD
#     scatter_idx = 1  # num patches dim
#     gather_idx = 2  # head count dim
#     batch_idx = 0
#     dp_B, wp_nrow, ncol, sp_hc, hs = input_shard.shape
#     flat_shard = input_shard.flatten(1, 2)  # -> B, N_p/wp, hc/sp, hs
#     # first all2all sets up real workload where input's sequence is parallelized
#     s_shard = _SeqAllToAll.apply(sp_group, flat_shard, scatter_idx, gather_idx, batch_idx)
#     uly_out = _SeqAllToAll.apply(sp_group, s_shard, gather_idx, scatter_idx, batch_idx)
#     wp_shard = rearrange(
#         uly_out,
#         'B (wp_nrow ncol) sp_hc hs -> B wp_nrow ncol sp_hc hs',
#         wp_nrow=wp_nrow
#     )
#     wp_shifted = window_parallel_shift(wp_shard, (-shift_h, -shift_w), wp_group)
#     wp_rearranged = rearrange_for_fa(wp_shifted)
#     wp_q, wp_k, wp_v = [wp_rearranged*(c+1) for c in range(3)]  # arbitrary q, k, v
#     wp_out = F.scaled_dot_product_attention(wp_q, wp_k, wp_v)
#     gathered_out = gather_3D_shard(wp_out, device_mesh, dp_dim=0, wp_dim=1, sp_dim=2)
#     assert torch.allclose(out, gathered_out)

#     print_rank0('passed test_3D_fwd')

#     ## LOSS
#     log_tensor_slice('out', out)
#     log_tensor_slice('wp_out', wp_out)
#     log_tensor_slice('gathered_out', gathered_out)
#     label = torch.randn_like(out)
#     wp_nwin = wp_nrow * ncol // (w_h*w_w)

#     slice_row =  slice(wp_nwin*wp_rank, wp_nwin*(wp_rank+1))
#     slice_hc =  slice(sp_hc*sp_rank, sp_hc*(sp_rank+1))
#     slice_dp =  slice(dp_B*dp_rank, dp_B*(dp_rank+1))
#     wp_label = label[slice_dp, slice_row, slice_hc, :, :]
#     loss = F.mse_loss(label, out)
#     wp_loss = F.mse_loss(wp_label, wp_out)
#     wp_loss_tmp = wp_loss.detach().clone()
#     dist.all_reduce(wp_loss_tmp, op=dist.ReduceOp.SUM)
#     wp_loss_avg = wp_loss_tmp / dist.get_world_size()
#     print_rank0(f"loss: {loss}")
#     print_rank0(f"wp_loss_avg: {wp_loss_avg}")

#     ## BWD
#     loss.backward()
#     wp_loss.backward()
#     gathered_grad = gather_3D_shard(
#         input_shard.grad, device_mesh, dp_dim=0, wp_dim=1, sp_dim=3
#     )
#     assert torch.allclose(input.grad, gathered_grad), \
#         f'the diff norm is: {torch.norm(input.grad-gathered_grad)}'

#     print_rank0('passed test_3D_bwd')


if __name__ == "__main__":
    torch.manual_seed(42999)
    torch.set_default_dtype(torch.float32)
    logging.basicConfig(level=logging.INFO)

    # setup parallel settings for ulysses, WP
    input, input_shard, device_mesh, args = \
        setup(ulysses_degree=2, window_parallelism_degree=4)
    
    test_window_parallel_shift_and_rev_shift(input, input_shard, device_mesh, args)
    test_ulysses(input, input_shard, device_mesh, args)
    test_window_parallel_shift(input, input_shard, device_mesh, args)
    test_WP_bwd(input, input_shard, device_mesh, args)
    
    # test_WP_fwd_bwd(input, input_shard, device_mesh, args)
    # test_3D_fwd_bwd(input, input_shard, device_mesh, args)