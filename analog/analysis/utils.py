from typing import Dict, Optional, List, Tuple, Union
import torch


def synchronize_device_unflatten(
    src: Dict[str, Dict[str, torch.Tensor]],
    tgt: Dict[str, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
):
    """
    Synchronize the device of two tensor dicts.

    Args:
        src (Dict[str, Dict[str, torch.Tensor]]): Source tensors
        tgt (Dict[str, Dict[str, torch.Tensor]]): Target tensors
        device (Optional[torch.device]): Device to synchronize to
    """

    for module_name, module_dict in tgt.items():
        for log in module_dict.keys():
            if device is None:
                device = src[module_name][log].device
            tgt[module_name][log] = tgt[module_name][log].to(device=device)


def synchronize_device_flatten(
    src: torch.Tensor,
    tgt: torch.Tensor,
    device: Optional[torch.device] = None,
):
    """
    Synchronize the device of two tensor data structures.

    Args:
        src (Dict[str, Dict[str, torch.Tensor]]): Source tensors
        tgt (List[torch.Tensor]): Target tensors
        paths(List[Tuple[str, str]]): Flatten context paths for target tesnors.
        device (Optional[torch.device]): Device to synchronize to
    """
    if device is None:
        src_device = src.device
    tgt.to(device=src_device)


def synchronize_device(
    src: Union[Dict[str, Dict[str, torch.Tensor]], torch.Tensor],
    tgt: Union[Dict[str, Dict[str, torch.Tensor]], torch.Tensor],
    device: Optional[torch.device] = None,
):
    if isinstance(src, dict):
        assert isinstance(tgt, dict)
        synchronize_device_unflatten(src, tgt, device)
    else:
        assert isinstance(src, torch.Tensor) and isinstance(tgt, torch.Tensor)
        synchronize_device_flatten(src, tgt, device)
