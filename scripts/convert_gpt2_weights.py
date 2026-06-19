#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights to LLMs-from-scratch-go binary format."""

import struct
import numpy as np
from transformers import GPT2Model


def convert():
    print("Loading GPT-2 from HuggingFace...")
    hf = GPT2Model.from_pretrained("gpt2")
    state = hf.state_dict()
    n_layer = hf.config.n_layer
    print(f"Loaded GPT-2 ({n_layer} layers)")

    # Detect key prefix: transformers 4.x wraps in "transformer." module, 5.x+ doesn't
    hp = "transformer." if any(k.startswith("transformer.") for k in state) else ""

    tensors = {}
    keys = []

    def add(key, data):
        if data.ndim == 1:
            data = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            data = data.reshape(1, *data.shape)
        keys.append((key, *data.shape))
        tensors[key] = data.astype(np.float32)

    def add_emb(key, data):
        # Go stores embeddings as [N, 1, emb_dim], not [1, N, emb_dim]
        data = data.reshape(data.shape[0], 1, data.shape[1])
        keys.append((key, *data.shape))
        tensors[key] = data.astype(np.float32)

    # Embeddings
    add_emb("Embeddings.TokenEmbedding", state[f"{hp}wte.weight"].numpy())
    add_emb("Embeddings.PositionalEmbedding", state[f"{hp}wpe.weight"].numpy())

    for i in range(n_layer):
        hp_layer = f"{hp}h.{i}."

        # Split c_attn into Wq, Wk, Wv
        c_attn = state[f"{hp_layer}attn.c_attn.weight"].numpy()
        d_model = c_attn.shape[0]
        wq = c_attn[:, :d_model]
        wk = c_attn[:, d_model : 2 * d_model]
        wv = c_attn[:, 2 * d_model :]
        wq, wk, wv = [x.T for x in (wq, wk, wv)]

        add(f"Blocks.{i}.Attention.Wq.Weight", wq)
        add(f"Blocks.{i}.Attention.Wk.Weight", wk)
        add(f"Blocks.{i}.Attention.Wv.Weight", wv)

        # Output projection Wo
        wo = state[f"{hp_layer}attn.c_proj.weight"].numpy().T
        add(f"Blocks.{i}.Attention.Wo.Weight", wo)

        # LayerNorm 1
        add(f"Blocks.{i}.LN1.Gamma", state[f"{hp_layer}ln_1.weight"].numpy())
        add(f"Blocks.{i}.LN1.Beta", state[f"{hp_layer}ln_1.bias"].numpy())

        # FFN
        fc = state[f"{hp_layer}mlp.c_fc.weight"].numpy().T
        add(f"Blocks.{i}.FFN.Linear1.Weight", fc)

        proj = state[f"{hp_layer}mlp.c_proj.weight"].numpy().T
        add(f"Blocks.{i}.FFN.Linear2.Weight", proj)

        # LayerNorm 2
        add(f"Blocks.{i}.LN2.Gamma", state[f"{hp_layer}ln_2.weight"].numpy())
        add(f"Blocks.{i}.LN2.Beta", state[f"{hp_layer}ln_2.bias"].numpy())

    # Final LayerNorm
    add("FinalNorm.Gamma", state[f"{hp}ln_f.weight"].numpy())
    add("FinalNorm.Beta", state[f"{hp}ln_f.bias"].numpy())

    # Output projection (tied with token embeddings in GPT-2)
    add("OutputProj.Weight", state[f"{hp}wte.weight"].numpy())

    # Write binary
    keys.sort(key=lambda x: x[0])
    out_path = "gpt2_124M.bin"

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0x4C4C4D00))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", len(keys)))

        for key, d0, d1, d2 in keys:
            key_bytes = key.encode()
            f.write(struct.pack("<i", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<i", d0))
            f.write(struct.pack("<i", d1))
            f.write(struct.pack("<i", d2))
            tensors[key].tofile(f)

    total_params = sum(d0 * d1 * d2 for _, d0, d1, d2 in keys)
    print(f"Wrote {len(keys)} tensors, {total_params:,} total params")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    convert()
