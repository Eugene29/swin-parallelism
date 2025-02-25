from mpi4py import MPI
import torch
import torch.distributed as dist
if torch.xpu.is_available():
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch
from distributed_utils import *
import os
import time
import math
import matplotlib.pyplot as plt


def benchmark_send_recv(
    src: int,
    dst: int,
    msg_size: int = 5_000_000,
    send_group = None,
    recv_group = None,
):
    msg = torch.ones(msg_size) * dist.get_rank()
    out = torch.empty_like(msg)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dist.all_reduce(out, group=send_group)  # warmup
    dist.all_reduce(out, group=recv_group)  # warmup

    loc_world_size = get_device_count()
    ## send-recv 
    # 1. Run first half then second half to avoid deadlock (cyclical?)
    # 2. Assuming single-ring is only used for intra-node. Any cleaner way to do 
    # handle? 
    if loc_world_size == world_size:  # intra-node => (single-ring)
        first_send_ranks = range(world_size // 2)
        first_recv_ranks = range(1, world_size//2 + 1, 1)
    else:  # (inter-node)
        first_send_ranks = range(world_size//2)
        first_recv_ranks = range(loc_world_size, world_size//2 + loc_world_size)

        # ## Nic-aware and dead-lock proof send-recv
        # # Get first send/recv ranks
        # # Group ranks in to four quadrants to first perform send/recvs quadrant 
        # # 1, 3 and then 2, 4 second. 
        # first_send_ranks = []
        # first_recv_ranks = []
        # half_loc_ws = loc_world_size // 2
        # loc_first_half_ranks = range(0, half_loc_ws)
        # for global_rank in range(world_size):
        #     local_rank = global_rank % loc_world_size
        #     in_left_half = local_rank in loc_first_half_ranks 
        #     in_top_half = global_rank < world_size//2
        #     in_first_quad = in_left_half and in_top_half
        #     in_fourth_quad = not (in_left_half or in_top_half)
        #     if in_first_quad or in_fourth_quad:
        #         first_recv_rank = (global_rank+loc_world_size) % world_size
        #         first_send_ranks.append(global_rank)
        #         first_recv_ranks.append(first_recv_rank)

    # print_rank0(f"first_recv_ranks: {first_recv_ranks}", flush=True)
    # print_rank0(f"first_send_ranks: {first_send_ranks}", flush=True)
    # raise KeyboardInterrupt()

    dist.barrier()
    # strt = record_event()
    torch.xpu.synchronize()
    strt = time.perf_counter_ns()

    # Q. Why doesn't below work? 
    # recv_req = dist.irecv(out, src)
    # send_req = dist.isend(msg, dst)

    # recv_req.wait()
    # send_req.wait()

    # first batch send/recv
    if rank in first_send_ranks:
        send_req = dist.send(msg, dst, send_group)
    if rank in first_recv_ranks:
        recv_req = dist.recv(out, src, recv_group)
    # second batch send/recv
    if rank not in first_send_ranks:
        send_req = dist.send(msg, dst, send_group)
    if rank not in first_recv_ranks:
        recv_req = dist.recv(out, src, recv_group)
    # end = record_event()
    torch.xpu.synchronize()
    end = time.perf_counter_ns()

    ## Sanity check
    # print_in_order(f"msg: {msg}", flush=True)
    # print_in_order(f"out: {out}", flush=True)

    ## Calculate minimum bandwidth (gpbs)
    synchronize()
    time_taken_sec = calc_time(strt, end)
    bandwidth_per_rank_gbps = 16 * msg_size / time_taken_sec / 1e9  # giga = 1e9
    min_bandwidth = torch.tensor(bandwidth_per_rank_gbps, dtype=torch.float)
    max_bandwidth = torch.tensor(bandwidth_per_rank_gbps, dtype=torch.float)

    dist.all_reduce(min_bandwidth, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_bandwidth, op=dist.ReduceOp.MAX)
    # print_in_order(f"bandwidth_per_rank (gbps): {bandwidth_per_rank_gbps:.2f}")

    return min_bandwidth.cpu(), max_bandwidth.cpu()


def setup_intra_node_src_dst(rank, world_size):
    """Assuming you are only one node"""
    # assert get_device_count() == world_size, 'Only works on one node'
    dst = rank+1 if rank != world_size-1 else 0
    src = rank-1 if rank != 0 else world_size-1
    return src, dst


def setup_inter_node_src_dst(rank, world_size):
    """Assuming you are on multi-node"""
    local_world_size = get_device_count()  # local world size
    assert local_world_size < world_size, 'Only works on multi-node'

    desired_dst = rank + local_world_size
    desired_src = rank - local_world_size
    dst = desired_dst if desired_dst < world_size else desired_dst % world_size
    src = desired_src if desired_src >= 0 else world_size + desired_src
    return src, dst

def launch_process_group(method: str = 'mpiexec'):
    if method == 'mpiexec':  # MPIEXEC
        comm = MPI.COMM_WORLD  
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        dist.init_process_group(backend=get_backend(),
                                rank=rank,
                                world_size=world_size)
    else:  # fall back to TORCHRUN
        dist.init_process_group(backend=get_backend())

if __name__ == "__main__":
    # The goal of clean programming is not to write as least code, but to make 
    # coding easier and more readable. 

    launch_process_group('mpiexec')
    # launch_process_group('torchrun')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = get_device_count()
    local_rank = rank % local_world_size
    torch.set_default_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    ## Set up message and src/dst
    # src, dst = setup_intra_node_src_dst(rank, world_size)
    src, dst = setup_inter_node_src_dst(rank, world_size)
    print_in_order(f"src, dst: {src, dst}")

    ## Initialize individual groups for faster send/recv
    ## FIXME: Use the individual ccl group trick to get beter perf
    send_group, recv_group = None, None
    ## Intra-node
    # for i in range(world_size):
    #     src_  = i
    #     dst_ = i+1 if i+1 < world_size else 0
    #     group = dist.new_group([src_, dst_])
    #     if src_ == rank:
    #         send_group = group
    #     if dst_ == rank:
    #         recv_group = group
    # print(f"[rank, dst]: {[rank, dst]}", flush=True)
    # print(f"[src, rank]: {[src, rank]}", flush=True)
    # Inter-node
    # for i in range(world_size):
    #     group_src  = i
    #     group_dst = (i+local_world_size) % world_size
    #     group = dist.new_group([group_src, group_dst])
    #     if group_src == rank:
    #         send_group = group
    #     if group_dst == rank:
    #         recv_group = group
    #     print_rank0(f"group_src, group_dst: {group_src, group_dst}", flush=True)


    ## Experiment power of 2 message sizes
    lst_min_bandwidth = []
    lst_max_bandwidth = []
    lst_msg_size = [2**p for p in range(18, 32, 2)]
    # lst_msg_size = [2**12]
    lst_logged_size = [16 * m / 1e6 for m in lst_msg_size]
    for i in range(10):
        for msg_size in lst_msg_size:
            print_rank0(f'iter {i}, msg_size {msg_size}')
            min_, max_ = benchmark_send_recv(src, dst, msg_size, send_group, recv_group)
            if i == 9:
                lst_min_bandwidth.append(min_)
                lst_max_bandwidth.append(max_)

    print_rank0(f"lst_min_bandwidth (Gbps): {lst_min_bandwidth}", flush=True)
    print_rank0(f"lst_max_bandwidth (Gbps): {lst_max_bandwidth}", flush=True)

    ## Plot
    num_nodes = world_size // local_world_size
    plot_pth = f'plots/send_recv_n{num_nodes}.png'
    os.makedirs(os.path.dirname(plot_pth), exist_ok=True)
    plt.plot(lst_logged_size, lst_min_bandwidth, label='min')
    plt.plot(lst_logged_size, lst_max_bandwidth, label='max')
    plt.xlabel('message size (Mb)')
    plt.ylabel('bandwidth per rank (gpbs)')
    plt.title(f'inter-node ping bandwidth for cyclical send/recv ({num_nodes} nodes)')
    plt.legend()
    plt.savefig(plot_pth)

## FIXME: there is a divergence issue when I test more than 1 send/recv at a time, but doesn't occur when I benchmark only one send/recv at a time. 
## Tried only using half num NIC simultaenously such that it resolves the nic imbalance issue, but it doesn't show significant improvement? 