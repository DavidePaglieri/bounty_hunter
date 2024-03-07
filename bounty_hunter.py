import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datautils import get_loaders

from models import *
from utils import *


class Vis:
    """Visualization util for a single layer"""

    def __init__(self, layer):
        self.dev = layer.weight.device
        self.columns = layer.weight.data.shape[0]
        self.output = torch.zeros((2048, self.columns), device=self.dev)
        self.nsamples = 0

    def normalize_output(self):
        self.output /= self.nsamples
        self.output = torch.mean(self.output, 0)

    def save_layer_output(self, out):
        self.output += out
        self.nsamples += 1


@torch.no_grad()
def get_inputs(model, data_iterable, dev):

    layer_inputs = []
    attention_masks = []
    position_ids = []
    layer_input_kwargs = []

    class LayerHijacker(nn.Module):
        """hijack layer's forward pass to cache data"""

        def __init__(self, m, device):
            super().__init__()
            self.module = m
            self.data_device = device

        def forward(self, inp=None, **kwargs):
            if (
                inp is None
            ):  # some models use all key-value arguments in forward pass call
                for kwarg_name in ["hidden_states"]:
                    if kwarg_name in kwargs:
                        inp = kwargs[kwarg_name]
                        break
            layer_inputs.append(move_to_device(inp, self.data_device))

            if kwargs["attention_mask"] is not None:
                attention_masks.append(kwargs["attention_mask"].to(self.data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to_device(pos_ids, self.data_device))
            one_kwargs = {}
            for (
                k,
                v,
            ) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to_device(v, self.data_device)
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

    model_class = CAUSAL_LM_MODEL_MAP[model.config.model_type.lower()]()
    layers = get_layers(model, model_class.layers_block_name)

    cur_layer_device = get_device(layers[0])
    # layers[0] = layers[0].to(dev)

    # get inputs for first layer
    layers[0] = LayerHijacker(layers[0], dev)
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(cur_layer_device))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(cur_layer_device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    layer_attention_mask = move_to_device(attention_masks[0], dev)
    additional_layer_inputs = {"attention_mask": layer_attention_mask}
    layer_position_ids = (
        None if not position_ids else move_to_device(position_ids[0], dev)
    )
    if layer_position_ids is not None:
        additional_layer_inputs["position_ids"] = layer_position_ids
    for k, v in layer_input_kwargs[0].items():
        additional_layer_inputs[k] = nested_move_to_device(v, dev)

    return layer_inputs, additional_layer_inputs


@torch.no_grad()
def visualize(model, dataloader, args, device):
    print("\nSaving visualizations ...")

    model.config.use_cache = False
    dtype = next(iter(model.parameters())).dtype

    # CAN TRY TO GIVE IT CPU HERE TO OFFLOAD MEMORY!
    inps, forward_args = get_inputs(model, dataloader, "cpu")
    outs = torch.zeros((args.nsamples, *inps[0].shape), dtype=dtype, device=device)

    output_attn_o_proj = torch.empty((0, model.config.hidden_size), device="cpu")
    output_mlp_down_proj = torch.empty((0, model.config.hidden_size), device="cpu")

    outputs = {
        "self_attn.o_proj": output_attn_o_proj,
        "mlp.down_proj": output_mlp_down_proj,
    }

    model_class = CAUSAL_LM_MODEL_MAP[model.config.model_type.lower()]()
    layers = get_layers(model, model_class.layers_block_name)
    for i in range(len(layers)):

        layer_dev_original = next(layers[i].parameters()).device
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        all_sublayers = find_sublayers(layer)

        sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}
            vis_handlers = {}
            for sublayer_name in subset:
                if sublayer_name in ["self_attn.o_proj", "mlp.down_proj"]:
                    vis_handlers[sublayer_name] = Vis(subset[sublayer_name])

            def save_output(name):
                def save(_, inp, out):
                    vis_handlers[name].save_layer_output(out[0].data)

                return save

            vis_handles = []
            for sublayer_name in subset:
                if sublayer_name in ["self_attn.o_proj", "mlp.down_proj"]:
                    vis_handles.append(
                        subset[sublayer_name].register_forward_hook(
                            save_output(sublayer_name)
                        )
                    )
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].to(layer_dev), **forward_args)[0]
                outs[j].to("cpu")
            for h in vis_handles:
                h.remove()
            torch.cuda.empty_cache()

            for sublayer_name in subset:
                if sublayer_name in ["self_attn.o_proj", "mlp.down_proj"]:
                    vis_handlers[sublayer_name].normalize_output()
                    outputs[sublayer_name] = torch.cat(
                        (
                            outputs[sublayer_name],
                            vis_handlers[sublayer_name].output.unsqueeze(0).to("cpu"),
                        ),
                        0,
                    )
        for j in range(args.nsamples):
            outs_batch = layer(inps[j].to(layer_dev), **forward_args)[0]
            outs[j] = outs_batch
            outs[j] = outs[j].cpu()
        del outs_batch

        layers[i] = layer.to(layer_dev_original)
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    return outputs["self_attn.o_proj"], outputs["mlp.down_proj"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to model to load",
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Path to load quantized statistics."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    directory_path = "./languages"

    tokenizer_transformers = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True
    )

    def extract_alphabet(language):
        return language.split("_")[1].split(".")[0]

    alphabet_frequency = {}

    languages = os.listdir(directory_path)
    for language in languages:
        alphabet = extract_alphabet(language)
        if alphabet in alphabet_frequency:
            alphabet_frequency[alphabet] += 1
        else:
            alphabet_frequency[alphabet] = 1

    sorted_languages = sorted(
        languages, key=lambda x: alphabet_frequency[extract_alphabet(x)]
    )

    # Iterate over languages
    for idx, language in enumerate(sorted_languages):

        print("")
        print(f"We are using: {language}, {idx}/{len(sorted_languages)}")

        torch.cuda.empty_cache()
        file_path = os.path.join(directory_path, language)

        with open(file_path, "r") as f:
            text = f.read()
        f.close()
        tokens_transformers = tokenizer_transformers(text, return_tensors="pt")

        if hasattr(tokens_transformers, "input_ids"):
            tokens_transformers = tokens_transformers.input_ids

        tokens_transformers = tokens_transformers[0]

        count = 2048
        token_list = []
        for i in range(
            0, len(tokens_transformers) - len(tokens_transformers) % count, count
        ):
            token_list.append(tokens_transformers[i : i + count].unsqueeze(0))

        args.nsamples = len(token_list)
        torch.save(token_list, f"./tokens.pth")

        model = get_model(args.model_path, args.load).train(False)
        model = model.to(device)

        dataloader = get_loaders(
            "./tokens.pth",
            nsamples=args.nsamples,
            seed=42,
            model_path=args.model_path,
        )

        # Check availabel GPU memory
        print(torch.cuda.mem_get_info()[0])
        torch.cuda.empty_cache()
        import gc

        gc.collect()

        output_attn_o_proj, output_mlp_down_proj = visualize(
            model, dataloader, args, device
        )
        print(f"Output attn shape: {output_attn_o_proj.shape}")
        print(f"Output mlp shape: {output_mlp_down_proj.shape}")

        torch.save(output_attn_o_proj, f"./outputs/{language.strip('.txt')}_attn.pt")
        torch.save(output_mlp_down_proj, f"./outputs/{language.strip('.txt')}_mlp.pt")
