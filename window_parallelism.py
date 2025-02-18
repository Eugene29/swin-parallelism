import torch
import torch.distributed as dist
from torch import Tensor
from typing import Tuple, Any

class _WindowParallelism(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_partitioned_input: torch.tensor, 
        shift_size: list[int],
        wp_group: dist.ProcessGroup
    ) -> torch.tensor:
        r'''Shift the windows of SWIN transfromer in a distributd setting
        
            Args:
                row_partitioned_input: (B, n_row_patches/wp, n_col_patches, hc, hs)
        '''
        wp_rank = dist.get_rank(wp_group)  # FIXME: Verify that this returns the group rank?
        wp_world_size = dist.get_world_size(wp_group)

        ## Prepare row_partitioned_input for a send/recv 
        # -> [B, n_row_patches/wp/2, n_col_patches, hc, hs)] x2
        horizontal_shift_size, vertical_shift_size = shift_size
        send_slice = slice(0, vertical_shift_size)  # send top row portion
        remain_slice = slice(vertical_shift_size, None)  # store lower row portion
        send_row_input = row_partitioned_input[:, send_slice, :, :, :].contiguous()
        remain_row_input = row_partitioned_input[:, remain_slice, :, :, :].contiguous()
        
        ## Shift with send-recv ("single ring")
        dst = wp_rank-1 if wp_rank != 0 else wp_world_size-1  # send to previous rank
        src = wp_rank+1 if wp_rank != wp_world_size-1 else 0
        global_dst = dist.get_global_rank(wp_group, dst)
        global_src = dist.get_global_rank(wp_group, src)
        recv_row_input = torch.empty_like(send_row_input)
        send_req = dist.isend(send_row_input, global_dst)
        recv_req = dist.irecv(recv_row_input, global_src)
        send_req.wait()
        recv_req.wait()

        ## Reshape -> [B, n_row_patches/wp, n_col_patches, hc, hs)]
        # vertically concatenate remaining bottom row on top of recv'd row
        concat_row_input = torch.cat([remain_row_input, recv_row_input], dim=1)
        left_slice = slice(horizontal_shift_size, None)
        right_slice = slice(0, horizontal_shift_size)
        # shift horizontally to the left by shift size
        shifted_row_input = torch.cat([
            concat_row_input[:, :, left_slice, :, : ],
            concat_row_input[:, :, right_slice, :, : ]
        ], dim=2).contiguous()

        return shifted_row_input
    
    @staticmethod
    def backward():
        raise NotImplementedError

def window_parallel_shift(
    row_partitioned_input: torch.tensor, 
    shift_size: list[int],
    wp_group: dist.ProcessGroup = None,
) -> torch.tensor:
    return _WindowParallelism.apply(row_partitioned_input, shift_size, wp_group)


# def single_all_to_all(input, scatter_idx, gather_idx, group):
#     seq_world_size = dist.get_world_size(group)
#     inp_shape = list(input.shape)
#     inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
#     if scatter_idx < 2:
#         input_t = input.reshape(
#             [seq_world_size, inp_shape[scatter_idx]] + \
#             inp_shape[scatter_idx + 1:]
#         ).contiguous()
#     else:
#         ## First
#         # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
#         # B, s, 
#         input_t = input.reshape(
#             [-1, seq_world_size, inp_shape[scatter_idx]] + \
#             inp_shape[scatter_idx + 1:]
#         ).transpose(0, 1).contiguous()

#     output = torch.empty_like(input_t)
#     dist.all_to_all_single(output, input_t, group=group)

#     # if scattering the seq-dim, transpose the heads back to the original dimension
#     if scatter_idx < 2:
#         output = output.transpose(0, 2).contiguous()

#     return output.reshape(
#         inp_shape[: gather_idx] + \
#         [inp_shape[gather_idx] * seq_world_size,] + \
#         inp_shape[gather_idx + 1:]).contiguous()


# class _SeqAllToAll(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx, 
#         input: Tensor, 
#         scatter_idx: int, 
#         gather_idx: int,
#         group: dist.ProcessGroup = None, 
#     ) -> Tensor:
#         ctx.scatter_idx = scatter_idx
#         ctx.gather_idx = gather_idx
#         ctx.group = group

#         return single_all_to_all(input, scatter_idx, gather_idx, group)

#     @staticmethod
#     def backward(
#         ctx: Any, 
#         *grad_output: Tensor
#     ) -> Tuple[None, Tensor, None, None]:
#         return (None, _SeqAllToAll.apply(*grad_output, ctx.gather_idx, ctx.scatter_idx, 
#                                          ctx.group), None, None)


# class _UlyssesAll2All():
#     def forward(
#         ctx,
#         input: torch.tensor,
#         gather_idx: int,
#         scatter_idx: int,
#         sp_group: dist.ProcessGroup
#     ) -> torch.tensor:
#         '''Do an all2all that gathers and scatters based on gather_idx and 
#         scatter_idx. Typically used to gather sequence length and scatter 
#         attention heads or vice versa.
#         '''
#         sp_rank = dist.get_rank(sp_group)
#         sp_world_size = dist.get_world_size(sp_group)
#         input_dim = list(input.shape)

#         # Assuming x dim: (B, s, hc, hs)
#         assert input_dim[scatter_idx] % sp_world_size == 0, \
#             "uneven head or sequence length is not yet supported"
#         input_s = input.tensor_split(scatter_idx)
#         out_lst = [torch.empty_like for _ in input_s]
#         dist.all_to_all(out_lst, input_s, sp_group)
#         out = torch.cat(out_lst, )
        
#         ## More performant version below:
#         # # shard the scatter dim
#         # sharded_dim = list(x.shape)
#         # sharded_dim[scatter_idx] = sharded_dim[sharded_dim] // sp_world_size
#         # sharded_dim.insert(scatter_idx, sp_world_size)
#         # # output_dim[gather_idx] = output_dim[gather_idx] * sp_world_size
#         # # output_dim[scatter_idx] = output_dim[scatter_idx] // sp_world_size

#         # # permute since all_to_all_single scatters and gather along first dimension
#         # x_t = x.view(sharded_dim)
#         # x_t = x.permute()
#         # output = torch.empty_like(x)
#         # dist.all_to_all_single()

# def ulysses_all2all():
#     return _UlyssesAll2All.apply(ctx)


