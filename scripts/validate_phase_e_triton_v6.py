"""Compare the Phase E Triton operator with PyTorch on frozen v6 tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from qwen_int4.backends import torch_reference_linear
from qwen_int4.quantization import PackingFormat
from qwen_int4.triton_kernels import linear_canonical_w4g64_triton, triton_available
from qwen_int4.validation import tensor_metrics


V6_SHA256 = "131a020962efc4e236b17fbd7da02f4aa9914e4c88a07b647e8cfe47dcaecab6"
V6_METADATA = {
    "format_version": "qwen-int4-awq-v1",
    "group_size": "64",
    "quantization": "canonical_awq_w4a16_asymmetric",
    "packing": "canonical_low_nibble_k_even_high_nibble_k_odd",
    "zero_point": "canonical_signed_packed_per_group",
}
CASES = (
    ("model.layers.0.self_attn.v_proj", 1, torch.float16, 0.01, 0.0625),
    ("model.layers.0.self_attn.q_proj", 8, torch.bfloat16, 0.02, 0.25),
    ("model.layers.0.mlp.down_proj", 4, torch.float16, 0.01, 0.0625),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not triton_available():
        raise RuntimeError("frozen v6 Triton parity validation requires CUDA and Triton")
    if sha256_file(checkpoint) != V6_SHA256:
        raise ValueError("checkpoint SHA-256 does not match the frozen canonical v6 artifact")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        mismatch = {key: (value, metadata.get(key)) for key, value in V6_METADATA.items() if metadata.get(key) != value}
        if mismatch:
            raise ValueError(f"checkpoint metadata mismatch: {mismatch}")
        results = []
        for index, (prefix, rows, dtype, rtol, atol) in enumerate(CASES):
            qweight = handle.get_tensor(f"{prefix}.qweight").cuda()
            scales = handle.get_tensor(f"{prefix}.scales").cuda()
            zeros = handle.get_tensor(f"{prefix}.zeros").cuda()
            out_features, groups = scales.shape
            in_features = groups * 64
            torch.manual_seed(20260831 + index)
            activations = torch.randn(rows, in_features, dtype=dtype, device="cuda")
            reference = torch_reference_linear(
                activations, qweight, scales, zeros, None,
                out_features=out_features, in_features=in_features, group_size=64,
                packing=PackingFormat.CANONICAL,
            )
            actual = linear_canonical_w4g64_triton(activations, qweight, scales, zeros)
            torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
            results.append({
                "layer": prefix,
                "shape_mnk": [rows, out_features, in_features],
                "activation_dtype": str(dtype).removeprefix("torch."),
                "rtol": rtol,
                "atol": atol,
                "metrics": tensor_metrics(reference, actual).to_dict(),
            })
    print(json.dumps({"checkpoint_sha256": V6_SHA256, "status": "passed", "cases": results}, indent=2))


if __name__ == "__main__":
    main()
