import torch
import torch.distributed as dist
# TODO: how to get this import to work inside test/
from distributed_utils import (
    print_in_order, print_rank0, get_backend
)


def benchmark_send_recv():
    ## Init distributed backend
    dist.init_process_group(backend=get_backend())
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    print_in_order(f"RANK: {RANK}")
    print_rank0(f"WORLD_SIZE: {WORLD_SIZE}\n")

    ## Set up message and src/dst
    msg_size = int(5e6)
    msg = torch.ones(msg_size) * RANK
    out = torch.empty_like(msg)
    dst = RANK+1 if RANK != WORLD_SIZE-1 else 0
    src = RANK-1 if RANK != 0 else WORLD_SIZE-1

    ## send-recv
    import time
    strt = time.perf_counter_ns()
    send_req = dist.isend(msg, dst)
    recv_req = dist.irecv(out, src)
    send_req.wait()
    recv_req.wait()
    end = time.perf_counter_ns()

    ## Calculate bandwidth
    time_taken = end - strt
    bandwidth_per_rank = 32 * msg_size / time_taken  # bandwidth = msg_size / time ?
    print_in_order(f"bandwidth_per_rank (gbps): {bandwidth_per_rank:.2f}")

if __name__ == "__main__":
    benchmark_send_recv()