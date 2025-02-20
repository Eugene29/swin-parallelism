import torch
import torch.distributed as dist

def get_backend():
    if torch.cuda.is_available():
        return 'nccl'
    if torch.xpu.is_available():
        return 'ccl'
    return 'gloo'


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
            print(f"{RANK}: {msg}", flush=True)
    else:
        print(msg)


def get_device_type():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.xpu.is_available():
        return 'xpu'
    return 'cpu'

def get_device_count():
    if torch.cuda.is_available():
        return torch.cuda.get_device_count()
    if torch.xpu.is_available():
        return torch.xpu.get_device_count()
    return dist.get_world_size()