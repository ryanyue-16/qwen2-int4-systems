"""Backend-dispatched quantized linear operator."""

from __future__ import annotations

import torch
from torch import nn

from .backends import SUPPORTED_BACKENDS, lpu_functional_linear, torch_reference_linear
from .quantization import PackingFormat


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 128,
        bias: bool = False,
        backend: str = "torch",
        packing: PackingFormat | str = PackingFormat.LEGACY_SIGNED_ADD,
    ) -> None:
        super().__init__()
        if in_features % group_size:
            raise ValueError(f"{in_features=} must be divisible by {group_size=}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.backend = "torch"
        self.packing = PackingFormat(packing)

        # Zero initialization makes an accidental partial load deterministic.
        self.register_buffer("qweight", torch.zeros(out_features * in_features // 2, 1, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(out_features, self.num_groups, dtype=torch.float16))
        self.register_buffer(
            "zeros",
            torch.zeros((out_features * self.num_groups + 1) // 2, 1, dtype=torch.uint8),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.set_backend(backend)

    def set_backend(self, backend: str) -> None:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {SUPPORTED_BACKENDS}")
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        implementation = {
            "torch": torch_reference_linear,
            "lpu": lpu_functional_linear,
        }[self.backend]
        return implementation(
            x,
            self.qweight,
            self.scales,
            self.zeros,
            self.bias,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
            packing=self.packing,
        )
