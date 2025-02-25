# TODO: get this import to work inside test/
# TODO: Change it to explicit imports
from distributed_utils import *
from window_parallelism import window_parallel_shift
from ulysses import _SeqAllToAll

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from einops import rearrange
from functools import partial

import os
import numpy as np
import logging
from collections import namedtuple


def log_tensor_slice(tensor_name, tensor):
    """log a tensor slice on rank 0
    Args:
        tensor: [B, nrow_patch, ncol_patch, hc, hs]
    """
    if dist.get_rank() == 0:
        logging.info(f"{tensor_name}:\n{tensor[0, :, :, 0, 0].long()}\n\n")


def setup(
    ulysses_degree=1, 
    window_parallelism_degree=1, 
):
    r'''Initialize comm groups. Create and return both row sharded and head count sharded 
    toy data.
    
    Returns:
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    '''
    ## Initialize 3D Device Mesh
    dist.init_process_group(backend=get_backend())  # Torchrun
    world_size = dist.get_world_size()
    SP, WP = (ulysses_degree, window_parallelism_degree)
    DP = world_size // SP // WP
    assert world_size == (DP*SP*WP), 'Your parallel degrees are likely wrong'
    device_mesh = init_device_mesh(device_type=get_device_type(), 
                                   mesh_shape=[DP, WP, SP],
                                   mesh_dim_names=['DP','WP','SP'])
    sp_group = device_mesh.get_group('SP')
    wp_group = device_mesh.get_group('WP')
    wp_rank = dist.get_rank(wp_group)
    sp_rank = dist.get_rank(sp_group)

    ## Set Image Dimensions
    B, hc, hs = (2, SP, 2)
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
    dtype = torch.get_default_dtype()
    input = torch.arange(np.prod(input_dim), dtype=dtype).view(input_dim)  

    ## Shard Image
    # Shard input by row -> B, n_patch_row/wp, n_patch_col, hc, hs
    assert nrow % WP == 0, 'num row windows must be divisible by WP'
    row_chunk = nrow // WP * win_h  # num row patches per WP
    row_slice = slice(row_chunk*wp_rank, row_chunk*(wp_rank+1))
    input_row_shard = input[:, row_slice, :, :, :]
    # Shard input by head count -> B, n_patch_row/wp, n_patch_col, hc/sp, hs
    assert (row_chunk*ncol_patch) % SP == 0, 'uneven head ulysses is not supported'
    assert hc % SP == 0, 'uneven head ulysses is not supported'
    hc_chunk = hc // SP  # num head count per SP(ulysses)
    hc_slice = slice(hc_chunk*sp_rank, hc_chunk*(sp_rank+1))
    input_shard = input_row_shard[:, :, :, hc_slice, :]

    log_tensor_slice('input', input)
    log_tensor_slice('input_row_shard', input_row_shard)
    log_tensor_slice('input_shard', input_shard)

    return input, input_shard, device_mesh, args


def test_window_parallel_shift(input, input_shard, device_mesh, args):
    """Test window parallel shift against normal shift (roll)
    Args:
        input: B, n_patch_row, n_patch_col, hc, hs
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    """

    neg_shift_size = (-args.shift_h, -args.shift_w)
    wp_group = device_mesh['WP'].get_group()
    sp_group = device_mesh['SP'].get_group()
    WP = dist.get_world_size(wp_group)
    SP = dist.get_world_size(sp_group)

    # Normal shift (roll)
    shifted = torch.roll(input, neg_shift_size, dims=(1,2))  # dim: nrow, ncol

    # Window Parallel Shift
    wp_shifted = window_parallel_shift(input_shard, neg_shift_size, wp_group=wp_group)

    ## Compare (by gathering parallel input)
    # gather along WP
    lst_wp_shard = [torch.empty_like(wp_shifted) for _ in range(WP)]
    dist.all_gather(lst_wp_shard, wp_shifted, group=wp_group)
    hc_shard = torch.cat(lst_wp_shard, dim=1)  # concat row dim
    # gather along SP
    lst_hc_shard = [torch.empty_like(hc_shard) for _ in range(SP)]
    dist.all_gather(lst_hc_shard, hc_shard, group=sp_group)
    gathered_shifted = torch.cat(lst_hc_shard, dim=3)  # concat hc dim

    log_tensor_slice('shifted', shifted)
    log_tensor_slice('gathered_shifted', gathered_shifted)
    assert torch.allclose(shifted, gathered_shifted)
    print_rank0("passed 'test_window_parallel_shift'")


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


def fwd_bwd_3D(input, input_shard, device_mesh, args):
    """Roughly simulate the workload of SWIN transformer and check the functionality of 
    fwd+bwd
    
    Args:
        input: B, n_patch_row, n_patch_col, hc, hs
        input_shard: B, n_patch_row/wp, n_patch_col, hc/sp, hs
    """

    ## fwd go brr
    # shift shift!!
    pass


if __name__ == "__main__":
    torch.manual_seed(42999)
    torch.set_default_dtype(torch.bfloat16)
    logging.basicConfig(level=logging.INFO)

    # setup parallel settings for ulysses, WP
    input, input_shard, device_mesh, args = \
        setup(ulysses_degree=4, window_parallelism_degree=3)
    
    test_window_parallel_shift(input, input_shard, device_mesh, args)
    test_window_parallel_shift_and_rev_shift(input, input_shard, device_mesh, args)
    test_ulysses(input, input_shard, device_mesh, args)


    ## TODO: How to get the tests to all run at once? 
    # test_intranode_WP_shift()
    # test_WP_shift_and_rev_shift()

    # test_intranode_window_parallel_fwd()
    # test_internode_window_parallel_shift()
    # test_ulysses()
