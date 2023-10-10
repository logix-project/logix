import torch.distributed as dist


def move_to_cpu(data):
    data = data.cpu()

def get_world_size(group=None):
    if dist.is_initialized():
        return dist.get_rank(group)
    else:
        return 0