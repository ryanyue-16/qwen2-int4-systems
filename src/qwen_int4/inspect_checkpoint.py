"""Metadata-only checkpoint inspection; does not require loading 1.1 GB into RAM."""

from __future__ import annotations

import argparse

import numpy as np
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="model.safetensors")
    args = parser.parse_args()

    with safe_open(args.checkpoint, framework="numpy", device="cpu") as handle:
        keys = list(handle.keys())
        qkeys = [key for key in keys if key.endswith(".qweight")]
        skeys = [key for key in keys if key.endswith(".scales")]
        zkeys = [key for key in keys if key.endswith(".zeros")]
        nonzero_zeros = [key for key in zkeys if np.any(handle.get_tensor(key))]
        layers = sorted(
            {int(key.split(".")[2]) for key in keys if key.startswith("model.layers.")}
        )

        print(f"checkpoint: {args.checkpoint}")
        print(f"tensors: {len(keys)}")
        print(f"layers: {len(layers)} ({layers[0]}..{layers[-1]})")
        print(f"quantized linears: qweight={len(qkeys)}, scales={len(skeys)}, zeros={len(zkeys)}")
        print(f"non-zero zero-point tensors: {len(nonzero_zeros)}")
        print(f"metadata: {handle.metadata()}")

        metadata = handle.metadata() or {}
        if metadata.get("format_version") == "qwen-int4-v1":
            print(f"declared packing: {metadata.get('packing')}")
            print("format is versioned; validate weights using quantization-report.json/HF comparison")

        sample = handle.get_tensor(qkeys[0]).reshape(-1)
        low = sample & 0x0F
        high = sample >> 4
        corrected_high = (high + (low >= 8).astype(np.uint8)) & 0x0F
        low_hist = np.bincount(low, minlength=16)
        raw_distance = int(np.abs(np.bincount(high, minlength=16) - low_hist).sum())
        corrected_distance = int(np.abs(np.bincount(corrected_high, minlength=16) - low_hist).sum())
        print(f"packing sample: {qkeys[0]}")
        print(f"high/low histogram L1: raw={raw_distance}, borrow-corrected={corrected_distance}")
        if metadata.get("format_version") == "qwen-int4-v1":
            return
        if corrected_distance < raw_distance:
            print("statistical packing hint: legacy_signed_add")
            print("warning: histogram evidence alone does not validate weight layout")
        else:
            print("statistical packing hint: canonical")
            print("warning: verify against original BF16 weights before inference")


if __name__ == "__main__":
    main()
