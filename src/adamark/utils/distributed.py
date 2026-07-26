import torch
import torch.distributed as dist


def ddp_mean(value: float, device, world_size: int) -> float:
    """Average a scalar across all DDP ranks"""

    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()
