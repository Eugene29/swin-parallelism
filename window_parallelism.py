import torch
import torch.distributed as dist
from torch import Tensor
from typing import Tuple, Any, Iterable
from distributed_utils import *

class _WindowParallelism(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_shard: Tensor, 
        shift_size: Iterable[int],
        wp_group: dist.ProcessGroup
    ) -> Tensor:
        r"""Shift the windows of SWIN Transformer by the shift-size where windows are parallelized across GPUs or nodes.

        Note:
            We follow the shift_size syntax of torch.roll()
        Args:
            input_shard: (B, n_row_patches/wp, n_col_patches, hc/sp, hs)
        """
        ctx.shift_size = shift_size
        ctx.wp_group = wp_group
        wp_rank = dist.get_rank(ctx.wp_group)
        wp_world_size = dist.get_world_size(ctx.wp_group)
        shift_h, shift_w = shift_size
        slice_top = slice(0, shift_h)
        slice_bot = slice(shift_h, None)
        top_shard = input_shard[:, slice_top, :, :, :].contiguous()
        bot_shard = input_shard[:, slice_bot, :, :, :].contiguous()
        wp_next_rank = wp_rank+1 if wp_rank != wp_world_size-1 else 0
        wp_prev_rank = wp_rank-1 if wp_rank != 0 else wp_world_size-1

        ## Vertical Shift: send/recv based on shift_size
        if shift_h < 0:  # shift up
            send_shard, remain_shard = (top_shard, bot_shard)
            src = dist.get_global_rank(wp_group, wp_next_rank)
            dst = dist.get_global_rank(wp_group, wp_prev_rank)
        else:  # shift down
            send_shard, remain_shard = (bot_shard, top_shard)
            src = dist.get_global_rank(wp_group, wp_prev_rank)
            dst = dist.get_global_rank(wp_group, wp_next_rank)
        # send in two batches to avoid deadlock
        first_send_ranks = range(wp_world_size//2)
        first_recv_ranks = range(1, wp_world_size//2 + 1)
        recv_shard = torch.empty_like(send_shard)
        # First batch
        # TODO: create individual group per send/recv for a speed up (aurora only?)
        if wp_rank in first_send_ranks:
            dist.send(send_shard, dst, group=wp_group)  
        if wp_rank in first_recv_ranks:
            dist.recv(recv_shard, src, group=wp_group)
        # Second batch
        if wp_rank not in first_send_ranks:  
            dist.send(send_shard, src, group=wp_group)
        if wp_rank not in first_recv_ranks:
            dist.recv(recv_shard, dst, group=wp_group)
        # fuse recv'd tensor and remain tensor
        if shift_h < 0: 
            concat_shard = torch.cat([remain_shard, recv_shard], dim=1)  # concat nrow dim
        else:
            concat_shard = torch.cat([recv_shard, remain_shard], dim=1)

        ## Horizontal Shift: concatenate and reshape
        slice_left = slice(0, -shift_w)
        slice_right = slice(-shift_w, None)
        # Works agnostically of sign of shift_w for horizontal case
        out = torch.cat(
            [concat_shard[:, :, slice_right, :, :], 
            concat_shard[:, :, slice_left, :, :]], dim=2
        ).contiguous()  # concat across ncol dim
        
        return out

    
    @staticmethod
    def backward(
        ctx,
        output_grads: Tensor, 
    ) -> Tuple[None, Tensor, None, None]:
        rev_shift_size = (-ctx.shift_size[0], -ctx.shift_size[1])
        return (
            _WindowParallelism.apply(output_grads, rev_shift_size, ctx.wp_group), 
            None, 
            None
        )


def window_parallel_shift(
    input_shard: Tensor, 
    shift_size: Iterable[int],
    wp_group: dist.ProcessGroup = None,
) -> Tensor:
    return _WindowParallelism.apply(input_shard, shift_size, wp_group)