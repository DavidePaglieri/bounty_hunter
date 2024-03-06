import torch
import torch.nn as nn
from typing import Optional, Union

            
def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device

def move_to_device(obj: Optional[Union[torch.Tensor, nn.Module]], device: torch.device):
    if obj is None:
        return obj
    else:
        if get_device(obj) != device:
            obj = obj.to(device)
        return obj

def nested_move_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to_device(e, device) for e in v])
    else:
        return v
    

