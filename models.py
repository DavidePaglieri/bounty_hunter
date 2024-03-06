import transformers
import torch
import torch.nn as nn



def find_sublayers(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_sublayers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_layers(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module
        

def get_model_type(model_name):
    if model_name in CAUSAL_LM_MODEL_MAP:
        return CAUSAL_LM_MODEL_MAP[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in CAUSAL_LM_MODEL_MAP")


class GPT2CausalLM():
    layer_type = "GPT2Block"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.wpe", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"],
    ]


class MistralCausalLM():
    layer_type = "MistralDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


class GemmaCausalLM():
    layer_type = "GemmaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


class LlamaCausalLM():
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


CAUSAL_LM_MODEL_MAP = {
    "gpt2": GPT2CausalLM,
    "llama": LlamaCausalLM,
    "mistral": MistralCausalLM,
    "gemma": GemmaCausalLM,
}