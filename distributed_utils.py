import torch
import torch.distributed as dist

# class DistributedUtils():
def get_backend():
    if torch.cuda.is_available():
        return 'nccl'
    if torch.xpu.is_available():
        return 'ccl'
    return 'gloo'


# def initialize_distributed_backend():
    # dist.init_process_group(backend=get_backend())
    # RANK = dist.get_rank()
    # WORLD_SIZE = dist.get_world_size()
    # print_in_order(f"RANK: {RANK}")
    # print_rank0(f"WORLD_SIZE: {WORLD_SIZE}")


def print_in_order(msg):
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    for i in range(WORLD_SIZE):
        if RANK == i:
            print(f"{RANK}: {msg}")
        dist.barrier()


def print_rank0(msg):
    if dist.is_initialized():
        RANK = dist.get_rank()
        if RANK == 0:
            print(f"{RANK}: {msg}")
    else:
        print(msg)