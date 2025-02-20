import torch
import torch.distributed as dist
from torch import Tensor
from typing import Tuple, Any

class _WindowParallelism(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_partitioned_input: Tensor, 
        shift_size: list[int],
        wp_group: dist.ProcessGroup
    ) -> Tensor:
        r'''Shift the windows of SWIN transfromer in a distributd setting
        
            Args:
                row_partitioned_input: (B, n_row_patches/wp, n_col_patches, hc, hs)
        '''
        ctx.shift_size = shift_size
        ctx.wp_group = wp_group
        wp_rank = dist.get_rank(ctx.wp_group)  # FIXME: Verify that this returns the group rank?
        wp_world_size = dist.get_world_size(ctx.wp_group)

        ## Prepare row_partitioned_input for a send/recv 
        # -> [B, n_row_patches/wp/2, n_col_patches, hc, hs)] x2
        horizontal_shift_size, vertical_shift_size = shift_size


        if vertical_shift_size < 0:  # negative means upward
            send_slice = slice(0, vertical_shift_size)  # send top row portion
            remain_slice = slice(vertical_shift_size, None)  # store lower row portion
            dst = wp_rank-1 if wp_rank != 0 else wp_world_size-1  # send to previous rank
            src = wp_rank+1 if wp_rank != wp_world_size-1 else 0
        else:
            remain_slice = slice(0, vertical_shift_size)  # store top row portion
            send_slice = slice(vertical_shift_size, None)  # send lower row portion
            dst = wp_rank+1 if wp_rank != wp_world_size-1 else 0  # send to next rank
            src = wp_rank-1 if wp_rank != 0 else wp_world_size-1  

        send_row_input = row_partitioned_input[:, send_slice, :, :, :].contiguous()
        remain_row_input = row_partitioned_input[:, remain_slice, :, :, :].contiguous()
        ## Shift with send-recv ("single ring")
        ## FIXME: Probably need to handle hangs by sending it in 2 rounds. 
        global_dst = dist.get_global_rank(ctx.wp_group, dst)
        global_src = dist.get_global_rank(ctx.wp_group, src)
        recv_row_input = torch.empty_like(send_row_input)
        # HACK: if vertical shift is neg -> flip the send recv op
        # TODO: Find a more interpretable implementation?
        send_req = dist.isend(send_row_input, global_dst)
        recv_req = dist.irecv(recv_row_input, global_src)
        send_req.wait()
        recv_req.wait()

        ## Reshape -> [B, n_row_patches/wp, n_col_patches, hc, hs)]
        # vertically concatenate remaining bottom row on top of recv'd row
        if vertical_shift_size < 0:
            concat_row_input = torch.cat([remain_row_input, recv_row_input], dim=1)
        else:
            concat_row_input = torch.cat([recv_row_input, remain_row_input], dim=1)

        left_slice = slice(-horizontal_shift_size, None)
        right_slice = slice(0, -horizontal_shift_size)
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
    row_partitioned_input: Tensor, 
    shift_size: list[int],
    wp_group: dist.ProcessGroup = None,
) -> Tensor:
    return _WindowParallelism.apply(row_partitioned_input, shift_size, wp_group)