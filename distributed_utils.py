import torch
import torch.distributed as dist
import time

def get_backend():
    if torch.cuda.is_available():
        return 'nccl'
    if torch.xpu.is_available():
        return 'ccl'
    return 'gloo'


def get_device_type():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


def get_device_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.xpu.is_available():
        return torch.xpu.device_count()
    else:
        raise KeyboardInterrupt()
    

def print_in_order(msg, **kwargs):
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    for i in range(WORLD_SIZE):
        if RANK == i:
            print(f"{RANK}: {msg}")
        dist.barrier()


def print_rank0(msg, **kwargs):
    if dist.is_initialized():
        RANK = dist.get_rank()
        if RANK == 0:
            print(f"{RANK}: {msg}", flush=True)
    else:
        print(msg)
        

def synchronize():
    if torch.cuda.is_available():
        return torch.cuda.synchronize()
    if torch.xpu.is_available():
        return torch.xpu.synchronize()
    

def record_event():
    ## FIXaME: check if perf_counter_ns have the same metric as Event.record()
    if torch.cuda.is_available():
        event = torch.cuda.Event(enable_timing=True)
    elif torch.xpu.is_available(): 
        # TODO: how to use updated torch version with xpu? 
        event = torch.xpu.Event(enable_timing=True)
    else:
        return time.perf_counter_ns()
    event.record()

    return event


def calc_time(strt, end):
    if isinstance(strt, int):  
        # assuming int => time.perf_counter_ns is used
        return (end - strt) / 1e9  # 1 sec = 1e9 ns
    else:  # accelerator event in ms
        return strt.elapsed_time(end) / 1e3
    