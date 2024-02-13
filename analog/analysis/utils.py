from typing import Dict, Optional, List, Tuple, Union
import torch


def synchronize_device(
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
