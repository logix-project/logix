import torch
import functools
from torch.profiler import profile, ProfilerActivity


def memory_profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, profile_memory=True) as prof:
            result = func(*args, **kwargs)

        print(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage"
                if device.type == "cuda"
                else "self_cpu_memory_usage"
            )
        )
        return result

    return wrapper
