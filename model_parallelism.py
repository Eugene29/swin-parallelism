import torch
import torch.distributed as dist

class _WindowParallelism(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_partitioned_input: torch.tensor, 
        shift_size: list[int],
        wp_group: dist.ProcessGroup
    ) -> torch.tensor:
        r'''Shift the windows of SWIN transfromer in a distributd setting'''
        wp_rank = dist.get_rank(wp_group)  # FIXME: Verify that this returns the group rank?
        wp_world_size = dist.get_world_size(wp_group)

        ## Prepare row_partitioned_input for a send/recv 
        # -> [B, n_row_patches/wp/2, n_col_patches, hc, hs)] x2
        vertical_shift_size = shift_size[1]
        remain_slice = slice(vertical_shift_size, 2*vertical_shift_size)  # store lower half
        send_slice = slice(0, vertical_shift_size)  # send top half
        remain_row_input = row_partitioned_input[:, remain_slice, :, :, :].contiguous()
        send_row_input = row_partitioned_input[:, send_slice, :, :, :].contiguous()

        ## Shift with send-recv ("single ring")
        dst = wp_rank-1 if wp_rank != 0 else wp_world_size-1  # send to previous rank
        src = wp_rank+1 if wp_rank != wp_world_size-1 else 0
        recv_row_input = torch.empty_like(send_row_input)
        send_req = dist.isend(send_row_input, dst)
        recv_req = dist.irecv(recv_row_input, src)
        send_req.wait()
        recv_req.wait()

        ## Reshape -> [B, n_row_patches/wp, n_col_patches, hc, hs)]
        # vertically concatenate remaining row input on top of recv'd row input
        concat_row_input = torch.cat([remain_row_input, recv_row_input], dim=1)
        horizontal_shift_size = shift_size[1]
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

## TODO:
# class _UlyssesAll2All():
#     def forward(ctx, x, gather_idx, scatter_idx, sp_group):
#         '''Gather and scatter dimensions based on gather_idx and scatter_idx.
#         Typically used to gather sequence and scatter attention heads or vice versa.

#         '''
#         # Assuming x dim: (B, s, hc, hs)
#         sp_world_size = dist.get_world_size(sp_group)

#         input_dim = list(x.shape)
#         assert input_dim[scatter_idx] % sp_world_size == 0, \
#             "uneven head or sequence length is not yet supported"

#         # shard the scatter dim
#         sharded_dim = list(x.shape)
#         sharded_dim[scatter_idx] = sharded_dim[sharded_dim] // sp_world_size
#         sharded_dim.insert(scatter_idx, sp_world_size)
#         # output_dim[gather_idx] = output_dim[gather_idx] * sp_world_size
#         # output_dim[scatter_idx] = output_dim[scatter_idx] // sp_world_size

#         # permute since all_to_all_single scatters and gather along first dimension
#         x_t = x.view(sharded_dim)
#         x_t = x.permute()
#         output = torch.empty_like(x)
#         dist.all_to_all_single()


# class UlyssesAll2All():
#     @staticmethod
#     def __call__(ctx):
#         _UlyssesAll2All.apply(ctx)


