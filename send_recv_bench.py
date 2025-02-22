from mpi4py import MPI
import torch
import torch.distributed as dist
# TODO: how to get this import to work inside test/
from distributed_utils import *
import os
import time
import math
import matplotlib.pyplot as plt


def benchmark_send_recv(
    src: int,
    dst: int,
    msg_size: int = 5_000_000,
):
    msg = torch.randn(msg_size) * dist.get_rank()
    out = torch.empty_like(msg)

    dist.all_reduce(out)  # warmup

    loc_world_size = get_device_count()
    ## send-recv 
    # Assuming single-ring is only used for intra-node. Any cleaner way to do 
    # handle? 
    # Run first half then second half to avoid deadlock (cyclical?)
    if loc_world_size == world_size:  # (single-ring)
        first_send_ranks = range(world_size // 2)
        first_recv_ranks = range(1, world_size//2 + 1, 1)
        if rank in first_send_ranks:
            send_req = dist.send(msg, dst)
        if rank in first_recv_ranks:
            recv_req = dist.recv(out, src)

        if rank not in first_send_ranks:
            send_req = dist.send(msg, dst)
        if rank not in first_recv_ranks:
            recv_req = dist.recv(out, src)
    else:  # (inter-node)
        first_recv_ranks = range(world_size//2)
        first_send_ranks = range(loc_world_size, world_size//2 + loc_world_size)

    synchronize()
    strt = time.perf_counter_ns()

    if rank in first_send_ranks:
        send_req = dist.send(msg, dst)
    if rank in first_recv_ranks:
        recv_req = dist.recv(out, src)
    if rank not in first_send_ranks:
        send_req = dist.send(msg, dst)
    if rank not in first_recv_ranks:
        recv_req = dist.recv(out, src)

    synchronize()
    end = time.perf_counter_ns()

    # ## Sanity check
    # print_in_order(f"msg: {msg}", flush=True)
    # print_in_order(f"out: {out}", flush=True)

    ## Calculate minimum bandwidth (gpbs)
    nano_seconds_taken = end - strt
    bandwidth_per_rank_gbps = 16 * msg_size / nano_seconds_taken  # n
    min_bandwidth = torch.tensor(bandwidth_per_rank_gbps, dtype=torch.float)
    max_bandwidth = torch.tensor(bandwidth_per_rank_gbps, dtype=torch.float)

    dist.all_reduce(min_bandwidth, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_bandwidth, op=dist.ReduceOp.MAX)
    print_in_order(f"bandwidth_per_rank (gbps): {bandwidth_per_rank_gbps:.2f}")
    # print_in_order(f"seconds_taken: {(nano_seconds_taken/1e9):.2f}")
    # print_rank0(f"min_bandwidth.cpu(), max_bandwidth.cpu(): {min_bandwidth.cpu(), max_bandwidth.cpu()}", flush=True)

    return min_bandwidth.cpu(), max_bandwidth.cpu()


def setup_intra_node_src_dst(rank, world_size):
    """Assuming you are only one node"""
    assert get_device_count() == world_size, 'Only works on one node'

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

    ## Set default device and dtype
    local_rank = rank % get_device_count()
    torch.set_default_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    ## Set up message and src/dst
    # src, dst = setup_intra_node_src_dst(rank, world_size)
    src, dst = setup_inter_node_src_dst(rank, world_size)
    # print_in_order(f"src, dst: {src, dst}")

    ## Experiment power of 2 message sizes
    lst_min_bandwidth = []
    lst_max_bandwidth = []
    # lst_msg_size = [2**p for p in range(18, 34, 2)]
    # lst_msg_size = [2**20, 2**30]
    lst_msg_size = [2**30]
    # lst_msg_size = [2**p for p in range(18, 28, 2)]
    lst_logged_size = [math.log10(m) for m in lst_msg_size]
    for msg_size in lst_msg_size:
        min_, max_ = benchmark_send_recv(src, dst, msg_size)
        # print_rank0(f"min_, max_: {min_, max_}", flush=True)
        # print_rank0(f"lst_msg_size[0]: {lst_msg_size[0]}", flush=True)
        lst_min_bandwidth.append(min_)
        lst_max_bandwidth.append(max_)

    # print(f"lst_min_bandwidth: {lst_min_bandwidth}", flush=True)
    # print(f"lst_max_bandwidth: {lst_max_bandwidth}", flush=True)
    # raise KeyboardInterrupt()

    ## Plot
    # plot_pth = 'plots/bandwidth.png'
    # os.makedirs(os.path.dirname(plot_pth), exist_ok=True)
    # plt.plot(lst_logged_size, lst_min_bandwidth, label='min')
    # plt.plot(lst_logged_size, lst_max_bandwidth, label='max')
    # plt.bar(lst_logged_size, lst_min_bandwidth, label='min')
    # plt.bar(lst_logged_size, lst_max_bandwidth, label='max')
    # plt.xlabel('message size (log10)')
    # plt.ylabel('bandwidth per rank (gpbs)')
    # plt.title('(Polaris) inter-node ping bandwidth for cyclical send/recv (2 nodes)')
    # plt.legend()
    # plt.savefig(plot_pth)

## FIXME: there is a divergence issue when I test more than 1 send/recv at a time, but doesn't occur when I benchmark only one send/recv at a time. 
## TODO:  Try first half, the other half simultaenously such that it resolves the nic imbalance issue. 