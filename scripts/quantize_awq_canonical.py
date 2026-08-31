"""Run canonical sequential self-AWQ calibration and export a new W4G128 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_int4.awq import (
    search_awq_scales,
    search_awq_scales_by_module_output,
    search_gqa_v_to_o_scales,
    search_groupwise_clipping,
)
from qwen_int4.calibration import ActivationCollector
from qwen_int4.evaluation import sha256_file
from qwen_int4.export import build_canonical_awq_export, save_canonical_awq_export
from qwen_int4.transforms import (
    expand_qwen2_gqa_scales,
    scale_linear_to_linear,
    scale_norm_to_linears,
)


def _linear_samples(samples: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        return samples[name]
    except KeyError as error:
        raise RuntimeError(f"missing calibration samples for {name}") from error


def _collect_block_samples(
    model,
    block,
    calibration_inputs: tuple[torch.Tensor, ...],
    max_cached_tokens: int,
    sampling_strategy: str,
    sample_seed: int,
    collect_attention_calls: bool,
):
    collector = ActivationCollector(
        max_cached_tokens=max_cached_tokens,
        sampling_strategy=sampling_strategy,
        sample_seed=sample_seed,
    )
    attention_collector = (
        AttentionCallCollector(max_cached_tokens) if collect_attention_calls else None
    )
    collector.attach(block)
    if attention_collector is not None:
        attention_collector.attach(block.self_attn)
    try:
        with torch.inference_mode():
            for input_ids in calibration_inputs:
                model(input_ids, use_cache=False)
        samples = collector.samples()
        statistics = collector.statistics()
    finally:
        if attention_collector is not None:
            attention_collector.detach()
        collector.detach()
    if any(
        module._forward_hooks or module._forward_pre_hooks
        for module in block.modules()
    ):
        raise RuntimeError("calibration hooks were not released")
    attention_calls = () if attention_collector is None else tuple(attention_collector.calls)
    return samples, statistics, attention_calls


def _clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    return value


def _to_device(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


class AttentionCallCollector:
    """Capture complete deterministic attention calls up to a token budget."""

    def __init__(self, max_cached_tokens: int) -> None:
        self.max_cached_tokens = max_cached_tokens
        self.cached_tokens = 0
        self.calls: list[tuple[torch.Tensor, dict]] = []
        self._handle = None

    def _hook(self, _module, args, kwargs) -> None:
        if self.cached_tokens >= self.max_cached_tokens:
            return
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise RuntimeError("attention call did not provide hidden states")
        call_kwargs = {key: value for key, value in kwargs.items() if key != "hidden_states"}
        positional_names = ("position_embeddings", "attention_mask", "past_key_values")
        for name, value in zip(positional_names, args[1:]):
            if name in call_kwargs:
                raise RuntimeError(f"attention argument {name} was provided twice")
            call_kwargs[name] = value
        self.calls.append((_clone_cpu(hidden_states), _clone_cpu(call_kwargs)))
        self.cached_tokens += hidden_states.numel() // hidden_states.shape[-1]

    def attach(self, attention) -> None:
        if self._handle is not None:
            raise RuntimeError("attention collector is already attached")
        self._handle = attention.register_forward_pre_hook(self._hook, with_kwargs=True)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _load_calibration_inputs(
    calibration: dict, tokenizer, *, root: Path, device: str, local_files_only: bool
) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Load deterministic unpadded calibration sequences from a local file or dataset."""

    source = calibration.get("source", "local_file")
    if source == "local_file":
        path = (root / calibration["path"]).resolve()
        if sha256_file(path) != calibration["sha256"]:
            raise ValueError("calibration checksum mismatch")
        max_tokens = int(calibration.get("max_sequence_length", 512))
        ids = tokenizer(
            path.read_text(encoding="utf-8"), return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[:, :max_tokens]
        if ids.shape[1] < 2:
            raise ValueError("calibration input must contain at least two tokens")
        return (ids.to(device),), {
            "source": source,
            "path": str(path.relative_to(root)),
            "sha256": calibration["sha256"],
            "samples": 1,
            "max_sequence_length": max_tokens,
        }
    if source != "huggingface_dataset":
        raise ValueError(f"unsupported calibration source: {source}")

    from datasets import DownloadConfig, load_dataset

    dataset = load_dataset(
        calibration["dataset"], calibration["config"], split=calibration["split"],
        revision=calibration["revision"],
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    text_column = calibration["text_column"]
    seed = int(calibration["seed"])
    samples = int(calibration["samples"])
    max_tokens = int(calibration["max_sequence_length"])
    dataset = dataset.filter(lambda row: bool(row[text_column].strip()))
    dataset = dataset.shuffle(seed=seed).select(range(samples))
    sequences: list[torch.Tensor] = []
    for row in dataset:
        ids = tokenizer(
            row[text_column], return_tensors="pt", padding=False,
            truncation=True, max_length=max_tokens,
            add_special_tokens=bool(calibration["add_special_tokens"]),
        ).input_ids
        if ids.shape[1] < 2:
            raise ValueError("a selected calibration sample produced fewer than two tokens")
        sequences.append(ids.to(device))
    if len(sequences) != samples:
        raise RuntimeError("calibration sample count drift")
    return tuple(sequences), {
        "source": source,
        "dataset": calibration["dataset"],
        "config": calibration["config"],
        "revision": calibration["revision"],
        "split": calibration["split"],
        "seed": seed,
        "samples": samples,
        "max_sequence_length": max_tokens,
        "add_special_tokens": bool(calibration["add_special_tokens"]),
    }


def _migrate_block_scales(
    block, samples, attention_calls, config: dict, device: str
) -> dict:
    quantization = config["quantization"]
    group_size = int(quantization["group_size"])
    grid_points = int(quantization["scale_search_grid_points"])
    duo_scaling = bool(quantization["duo_scaling"])
    attn = block.self_attn
    mlp = block.mlp
    transforms: dict[str, dict] = {}

    qkv = (attn.q_proj, attn.k_proj, attn.v_proj)
    qkv_objective = quantization.get("qkv_scale_objective", "linear_output")
    if qkv_objective == "attention_output":
        if not attention_calls:
            raise RuntimeError("attention-output search requires captured attention calls")
        attention_inputs = tuple(
            hidden_states.to(device=device, dtype=block.input_layernorm.weight.dtype)
            for hidden_states, _kwargs in attention_calls
        )
        attention_kwargs = tuple(
            _to_device(kwargs, device) for _hidden_states, kwargs in attention_calls
        )

        def forward_attention(index: int, hidden_states: torch.Tensor) -> torch.Tensor:
            return attn(hidden_states, **attention_kwargs[index])[0]

        qkv_result = search_awq_scales_by_module_output(
            attention_inputs,
            qkv,
            forward_attention,
            group_size=group_size,
            grid_points=grid_points,
            duo_scaling=duo_scaling,
        )
    elif qkv_objective == "linear_output":
        qkv_result = search_awq_scales(
            _linear_samples(samples, "self_attn.q_proj"), qkv,
            group_size=group_size, grid_points=grid_points, duo_scaling=duo_scaling,
        )
    else:
        raise ValueError(f"unsupported QKV scale objective: {qkv_objective}")
    scale_norm_to_linears(block.input_layernorm, qkv, qkv_result.scales)
    transforms["input_layernorm_to_qkv"] = {
        "objective": qkv_objective,
        "ratio": qkv_result.ratio, "loss": qkv_result.loss,
        "identity_loss": qkv_result.identity_loss,
    }

    v_result = search_gqa_v_to_o_scales(
        _linear_samples(samples, "self_attn.o_proj"), attn.o_proj,
        num_attention_heads=attn.config.num_attention_heads,
        num_key_value_heads=attn.config.num_key_value_heads,
        group_size=group_size, grid_points=grid_points, duo_scaling=duo_scaling,
    )
    expanded = expand_qwen2_gqa_scales(
        v_result.scales,
        num_attention_heads=attn.config.num_attention_heads,
        num_key_value_heads=attn.config.num_key_value_heads,
    )
    scale_linear_to_linear(attn.v_proj, attn.o_proj, v_result.scales, expanded_target_scales=expanded)
    transforms["v_to_o"] = {
        "ratio": v_result.ratio, "loss": v_result.loss,
        "identity_loss": v_result.identity_loss,
    }

    gate_up = (mlp.gate_proj, mlp.up_proj)
    gate_up_result = search_awq_scales(
        _linear_samples(samples, "mlp.gate_proj"), gate_up,
        group_size=group_size, grid_points=grid_points, duo_scaling=duo_scaling,
    )
    scale_norm_to_linears(block.post_attention_layernorm, gate_up, gate_up_result.scales)
    transforms["post_attention_layernorm_to_gate_up"] = {
        "ratio": gate_up_result.ratio, "loss": gate_up_result.loss,
        "identity_loss": gate_up_result.identity_loss,
    }

    up_result = search_awq_scales(
        _linear_samples(samples, "mlp.down_proj"), (mlp.down_proj,),
        group_size=group_size, grid_points=grid_points, duo_scaling=duo_scaling,
    )
    scale_linear_to_linear(mlp.up_proj, mlp.down_proj, up_result.scales)
    transforms["up_to_down"] = {
        "ratio": up_result.ratio, "loss": up_result.loss,
        "identity_loss": up_result.identity_loss,
    }

    return {"transforms": transforms}


def _search_block_clipping(block, samples, config: dict) -> tuple[dict[str, torch.Tensor], dict]:
    quantization = config["quantization"]
    group_size = int(quantization["group_size"])
    clip_ratios = tuple(float(value) for value in quantization["clip_ratios"])
    clip_limits: dict[str, torch.Tensor] = {}
    clip_summary: dict[str, dict] = {}
    for name, module in block.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        result = search_groupwise_clipping(
            module.weight, _linear_samples(samples, name),
            group_size=group_size, ratios=clip_ratios,
        )
        clip_limits[name] = result.clip_max
        clip_summary[name] = {
            "mean_ratio": result.ratios.float().mean().item(),
            "fraction_clipped": (result.ratios < 1).float().mean().item(),
            "mean_loss": result.loss.mean().item(),
        }
    return clip_limits, {"clipping": clip_summary}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/awq-canonical-v2.json"))
    parser.add_argument(
        "--output",
        default=str(root / "artifacts/qwen2-1.5b-w4g128-awq-canonical-v2-gpu-r1"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-calibration-tokens", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--search-smoke-blocks",
        type=int,
        help="Run scale search for the first N blocks and exit without exporting weights.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("canonical self-AWQ calibration requires CUDA")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration = config["calibration"]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    model_config = config["model"]
    seed = int(calibration.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"],
        local_files_only=args.local_files_only,
    )
    calibration_inputs, calibration_provenance = _load_calibration_inputs(
        calibration, tokenizer, root=root, device=args.device,
        local_files_only=args.local_files_only,
    )
    invariant_input_ids = calibration_inputs[0]
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"], dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    with torch.inference_mode():
        baseline_logits = model(
            invariant_input_ids, use_cache=False
        ).logits.detach().float().cpu()

    all_clips: dict[str, torch.Tensor] = {}
    block_reports: list[dict] = []
    sampling_strategy = calibration.get("activation_sampling_strategy", "prefix")
    sample_seed = int(calibration.get("activation_sampling_seed", seed))
    qkv_objective = config["quantization"].get(
        "qkv_scale_objective", "linear_output"
    )
    blocks = tuple(model.model.layers)
    if args.search_smoke_blocks is not None:
        if not 1 <= args.search_smoke_blocks <= len(blocks):
            raise ValueError("search smoke block count is out of range")
        blocks = blocks[: args.search_smoke_blocks]
    for index, block in enumerate(blocks):
        samples, statistics, attention_calls = _collect_block_samples(
            model, block, calibration_inputs,
            int(calibration["max_cached_tokens_per_linear"]),
            sampling_strategy, sample_seed,
            qkv_objective == "attention_output",
        )
        search_samples = {
            name: value.to(device=args.device, dtype=torch.float32)
            for name, value in samples.items()
        }
        report = _migrate_block_scales(
            block, search_samples, attention_calls, config, args.device
        )
        recollect_for_clipping = bool(
            calibration.get("recollect_after_scale_migration", False)
        )
        if recollect_for_clipping:
            clipping_samples, clipping_statistics, _attention_calls = _collect_block_samples(
                model, block, calibration_inputs,
                int(calibration["max_cached_tokens_per_linear"]),
                sampling_strategy, sample_seed,
                False,
            )
            clipping_search_samples = {
                name: value.to(device=args.device, dtype=torch.float32)
                for name, value in clipping_samples.items()
            }
        else:
            clipping_samples, clipping_statistics = samples, statistics
            clipping_search_samples = search_samples
        clips, clip_report = _search_block_clipping(
            block, clipping_search_samples, config
        )
        report.update(clip_report)
        prefix = f"model.layers.{index}."
        all_clips.update({prefix + name: value for name, value in clips.items()})
        report["block_index"] = index
        report["scale_search_sampled_linears"] = sorted(samples)
        report["scale_search_observed_tokens"] = {
            name: value.count for name, value in statistics.items()
        }
        report["clipping_search_recollected_after_scale_migration"] = recollect_for_clipping
        report["activation_sampling_strategy"] = sampling_strategy
        report["activation_sampling_seed"] = sample_seed
        report["clipping_search_sampled_linears"] = sorted(clipping_samples)
        report["clipping_search_observed_tokens"] = {
            name: value.count for name, value in clipping_statistics.items()
        }
        block_reports.append(report)
        print(f"calibrated block {index + 1}/{len(blocks)}")

    if args.search_smoke_blocks is not None:
        print(json.dumps({
            "status": "search_smoke_complete",
            "blocks": block_reports,
        }))
        return

    with torch.inference_mode():
        transformed_logits = model(
            invariant_input_ids, use_cache=False
        ).logits.detach().float().cpu()
    transform_max_abs = (baseline_logits - transformed_logits).abs().max().item()
    transform_mean_abs = (baseline_logits - transformed_logits).abs().mean().item()
    baseline_mean_abs = baseline_logits.abs().mean().item()
    transform_relative_mean_abs = transform_mean_abs / max(baseline_mean_abs, 1e-8)
    transform_cosine = torch.nn.functional.cosine_similarity(
        baseline_logits.reshape(1, -1), transformed_logits.reshape(1, -1)
    ).clamp(max=1.0).item()
    if (
        not torch.isfinite(transformed_logits).all()
        or transform_cosine < 0.9999
        or transform_relative_mean_abs > 0.02
    ):
        raise RuntimeError(
            "AWQ scale migration failed invariant check: "
            f"max_abs={transform_max_abs}, mean_abs={transform_mean_abs}, "
            f"relative_mean_abs={transform_relative_mean_abs}, cosine={transform_cosine}"
        )

    export = build_canonical_awq_export(
        model,
        group_size=int(config["quantization"]["group_size"]),
        clip_max_by_layer=all_clips,
    )
    checkpoint, report_path = save_canonical_awq_export(export, output)
    run_report = {
        "schema_version": 1,
        "status": "generated_pending_quality_evaluation",
        "config": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "calibration": calibration_provenance,
        "transform_invariant_max_abs": transform_max_abs,
        "transform_invariant_mean_abs": transform_mean_abs,
        "transform_invariant_relative_mean_abs": transform_relative_mean_abs,
        "transform_invariant_cosine": transform_cosine,
        "blocks": block_reports,
        "checkpoint": str(checkpoint),
        "quantization_report": str(report_path),
    }
    (output / "calibration-report.json").write_text(
        json.dumps(run_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "transform_invariant_max_abs": transform_max_abs,
        "transform_invariant_mean_abs": transform_mean_abs,
        "transform_invariant_relative_mean_abs": transform_relative_mean_abs,
        "transform_invariant_cosine": transform_cosine,
    }))


if __name__ == "__main__":
    main()
