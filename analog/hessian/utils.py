import torch
import torch.nn.functional as F


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def extract_patches(x, kernel_size, stride, padding, groups):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).data
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    return torch.mean(
        x.reshape((x.size(0), x.size(1), x.size(2), groups, -1, x.size(4), x.size(5))),
        3,
    ).view(x.size(0), x.size(1), x.size(2), -1)
