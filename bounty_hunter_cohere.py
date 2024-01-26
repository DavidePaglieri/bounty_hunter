import os
import time

import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoTokenizer
import pickle

from datautils import get_loaders
from modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
)
from spqr_engine import SPQRUtil


class Vis:
    """Visualization util for a single layer"""

    def __init__(self, layer):
        self.dev = layer.weight.device
        self.columns = layer.weight.data.shape[1]
        self.output = torch.zeros((2048, 4096), device=self.dev)
        self.nsamples = 0

    def normalize_output(self):
        self.output /= self.nsamples
        self.output = torch.mean(self.output, 0)

    def save_layer_output(self, out):
        self.output += out
        self.nsamples += 1


try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    # print("catching inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = (
                model.model.decoder.embed_positions.to(dev)
            )
            if (
                hasattr(model.model.decoder, "project_in")
                and model.model.decoder.project_in
            ):
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.
    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    forward_arg_names = [
        "attention_mask",
    ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)
    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            emb_dev
        )
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args


@torch.no_grad()
def visualize(model, dataloader, args, device):
    print("\nSaving visualizations ...")

    inps, forward_args = get_inps(
        model, dataloader, args, dev="cpu" if args.offload_activations else device
    )
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    output_attn_o_proj = torch.empty((0, 4096), device="cpu")
    output_mlp_down_proj = torch.empty((0, 4096), device="cpu")
    print(model.config.model_type)
    print(device)

    layers = get_layers(model)
    for i in range(len(layers)):
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}
            vis_handlers = {}
            for sublayer_name in subset:
                if sublayer_name == "self_attn.out_proj" or sublayer_name == "fc2":
                    vis_handlers[sublayer_name] = Vis(subset[sublayer_name])
                if sublayer_name == "fc2":
                    vis_handlers[sublayer_name] = Vis(subset[sublayer_name])

            def save_output(name):
                def save(_, inp, out):
                    vis_handlers[name].save_layer_output(out[0].data)

                return save

            vis_handles = []
            for sublayer_name in subset:
                if sublayer_name == "self_attn.out_proj" or sublayer_name == "fc2":
                    vis_handles.append(
                        subset[sublayer_name].register_forward_hook(
                            save_output(sublayer_name)
                        )
                    )
                if sublayer_name == "fc2":
                    vis_handles.append(
                        subset[sublayer_name].register_forward_hook(
                            save_output(sublayer_name)
                        )
                    )
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
                outs[j].to("cpu")
            for h in vis_handles:
                h.remove()
            torch.cuda.empty_cache()

            for sublayer_name in subset:
                if sublayer_name == "self_attn.out_proj":
                    vis_handlers[sublayer_name].normalize_output()
                    output_attn_o_proj = torch.cat(
                        (
                            output_attn_o_proj,
                            vis_handlers["self_attn.out_proj"]
                            .output.unsqueeze(0)
                            .to("cpu"),
                        ),
                        0,
                    )
                if sublayer_name == "fc2":
                    vis_handlers[sublayer_name].normalize_output()
                    output_mlp_down_proj = torch.cat(
                        (
                            output_mlp_down_proj,
                            vis_handlers["fc2"].output.unsqueeze(0).to("cpu"),
                        ),
                        0,
                    )

        for j in range(args.nsamples):
            outs_batch = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
            outs[j] = outs_batch
            outs[j] = outs[j].cpu()
        del outs_batch

        layers[i] = layer.to(layer_dev_original)
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    return output_attn_o_proj, output_mlp_down_proj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="none",
        help="Dataset name [c4, pajama, refinedweb, none, etc.] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to load if specified. Deprecated",
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Path to load quantized statistics."
    )

    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=float("inf"),
        help="relative threshold for outliers; higher threshold = more outliers.",
    )
    parser.add_argument(
        "--simplified_outliers",
        action="store_true",
        help="do not perform leave-one-out evaluation when detecting outliers; works faster, but generally worse in perplexity",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Whether to use wandb or store locally."
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        nsamples = len(token_list)
        torch.save(token_list, f"./tokens.pth")

        model = get_model(args.model_path, args.load, args.dtype).train(False)

        dataloader = get_loaders(
            "./tokens.pth",
            nsamples=nsamples,
            seed=42,
            model_path=args.model_path,
        )
        output_attn_o_proj, output_mlp_down_proj = visualize(
            model, dataloader, args, device
        )

        torch.save(output_attn_o_proj, f"./outputs/{language}_attn.pt")
        torch.save(output_mlp_down_proj, f"./outputs/{language}_mlp.pt")
