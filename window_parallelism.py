import torch
import torch.distributed as dist
from torch import Tensor
from typing import Tuple, Any

class _WindowParallelism(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_shard: Tensor, 
        shift_size: list[int],
        wp_group: dist.ProcessGroup
    ) -> Tensor:
        r'''Shift the windows of SWIN transfromer in a distributd setting
        
        Args:
            input_shard: (B, n_row_patches/wp, n_col_patches, hc or hc/sp, hs)
        '''
        ctx.shift_size = shift_size
        ctx.wp_group = wp_group
        wp_rank = dist.get_rank(ctx.wp_group)
        wp_world_size = dist.get_world_size(ctx.wp_group)
        shift_h, shift_w = shift_size

        if shift_h < 0:  # send upward
            slice_send = slice(0, shift_h)  # send top row portion
            slice_store = slice(shift_h, None)  # store lower row portion
            dst = wp_rank-1 if wp_rank != 0 else wp_world_size-1  # send to prev rank
            src = wp_rank+1 if wp_rank != wp_world_size-1 else 0
        else:  # send downwards
            slice_send = slice(shift_h, None)  # send lower row portion
            slice_store = slice(0, shift_h)  # store top row portion
            dst = wp_rank+1 if wp_rank != wp_world_size-1 else 0  # send to next rank
            src = wp_rank-1 if wp_rank != 0 else wp_world_size-1  

        row_send = input_shard[:, slice_send, :, :, :].contiguous()
        row_store = input_shard[:, slice_store, :, :, :].contiguous()
        ## Shift with send-recv ("single ring")
        ## FIXME: Probably need to handle hangs by sending it in 2 rounds.
        ## TODO: add individual group trick for faster send/recv
        global_dst = dist.get_global_rank(ctx.wp_group, dst)
        global_src = dist.get_global_rank(ctx.wp_group, src)
        row_recv = torch.empty_like(row_send)
        send_req = dist.isend(row_send, global_dst)
        recv_req = dist.irecv(row_recv, global_src)
        send_req.wait()
        recv_req.wait()

        ## Reshape -> [B, n_row_patches/wp, n_col_patches, hc, hs)]
        # vertically concatenate remaining bottom row on top of recv'd row
        if shift_h < 0:
            concat_row_input = torch.cat([row_store, row_recv], dim=1)
        else:
            concat_row_input = torch.cat([row_recv, row_store], dim=1)

        left_slice = slice(-shift_w, None)
        right_slice = slice(0, -shift_w)
        # shift to the left by shift size
        shifted_row_input = torch.cat([
            concat_row_input[:, :, left_slice, :, : ],
            concat_row_input[:, :, right_slice, :, : ]
        ], dim=2).contiguous()

        return shifted_row_input
    
    @staticmethod
    def backward(
        ctx,
        output_grads: Tensor, 
    ) -> Tuple[None, Tensor, None, None]:
        reverse_shift_size = (-ctx.shift_size[0], -ctx.shift_size[1])
        return (
            _WindowParallelism.apply(output_grads, reverse_shift_size, ctx.wp_group), 
            None, 
            None
        )


def window_parallel_shift(
    input_shard: Tensor, 
    shift_size: list[int],
    wp_group: dist.ProcessGroup = None,
) -> Tensor:
    return _WindowParallelism.apply(input_shard, shift_size, wp_group)