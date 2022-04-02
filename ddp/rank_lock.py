import torch.distributed as dist


def rank0_first(fn):
    if not dist.is_initialized():
        return fn

    def warpper(*args, **kwargs):
        if dist.get_rank() == 0:
            fn(*args, **kwargs)
        dist.barrier()
        return fn(*args, **kwargs)

    return warpper
