from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


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


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = (
        skip  # replacing
    )
    try:
        yield
    finally:
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = saved_inits  # restoring


def get_model(model_path, seqlen=2048, dtype="auto"):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype
            or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        print("Loading pretrained model ...")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    model.seqlen = seqlen

    print("Model loaded sucessfully ...")

    return model
