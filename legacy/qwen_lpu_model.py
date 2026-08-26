import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

# -------------------- LPU emulate toggle --------------------
# 1 = emulate LPU MM dataflow: stream-on-K (group tiles), fp32 accumulate, PC at end
LPU_EMULATE = bool(int(os.environ.get("LPU_EMULATE", "1")))
LPU_RELU    = bool(int(os.environ.get("LPU_RELU", "0")))
LPU_OUT_DTYPE = os.environ.get("LPU_OUT_DTYPE", "bf16").lower()   # bf16|int8|fp8
LPU_SPARSE    = os.environ.get("LPU_SPARSE", "none").lower()      # none|4_8|2_4

# -------------------- utils --------------------

def _pc_postprocess(acc_fp32: torch.Tensor,
                    bias: torch.Tensor | None,
                    relu_en: bool,
                    out_dtype: str,
                    out_scale: torch.Tensor | None,
                    ref_dtype: torch.dtype):
    # Bias (PC block)
    if bias is not None:
        acc_fp32 = acc_fp32 + bias.to(acc_fp32.dtype)

    # ReLU (optional, PC block)
    if relu_en:
        acc_fp32 = torch.relu(acc_fp32)

    # Convert / Quantize (PC block)
    if out_dtype in ("bf16", "fp16", "float16"):
        return acc_fp32.to(ref_dtype)  # your model runs bf16 → keep identical numerics

    if out_dtype == "int8":
        assert out_scale is not None and out_scale.numel() == acc_fp32.shape[-1], \
            "int8 requires per-channel out_scale of shape [N]"
        y = acc_fp32 * out_scale.to(acc_fp32.dtype)          # scale to int8 range
        y = torch.clamp(torch.round(y), -128, 127).to(torch.int8)
        return y

    if out_dtype == "fp8":
        # Simple fp8 e5m2 emulation via scale → clamp → de/req
        # If your PyTorch has float8 dtypes you can swap this.
        assert out_scale is not None and out_scale.numel() == acc_fp32.shape[-1], \
            "fp8 requires per-channel out_scale of shape [N]"
        y = acc_fp32 * out_scale.to(acc_fp32.dtype)
        # clamp to plausible fp8 dynamic range (~[-448, 448]) as a guard
        y = torch.clamp(y, -448.0, 448.0)
        return y  # keep as fp32 (emulated). Replace with real fp8 if desired.

    # default: no conversion
    return acc_fp32.to(ref_dtype)

def _apply_sparse_select(a_tile: torch.Tensor,
                         w_tile: torch.Tensor,
                         mask_1d: torch.Tensor | None):
    """
    a_tile: [M, k_chunk], w_tile: [N, k_chunk], mask_1d: [k_chunk] bool
    Returns masked versions if mask provided; else originals.
    """
    if mask_1d is None or mask_1d.numel() == 0:
        return a_tile, w_tile
    idx = mask_1d.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        # avoid empty matmul; return zeros with correct shape
        return a_tile[:, :0], w_tile[:, :0]
    return a_tile[:, idx], w_tile[:, idx]

def load_quant_state_strict_subset(module: nn.Module, path: str, device="cpu"):
    ck = load_file(path, device=device)  # dict[str, Tensor]
    own = module.state_dict()
    new_sd = {}
    matched = ignored = shape_mismatch = 0
    ignored_keys = []
    mismatched_keys = []
    for k, v in ck.items():
        if k in own:
            if own[k].shape == v.shape:
                new_sd[k] = v.to(dtype=own[k].dtype)
                matched += 1
            else:
                new_sd[k] = own[k]
                shape_mismatch += 1
        else:
            ignored += 1
    for k, v in own.items():
        if k not in new_sd:
            new_sd[k] = v
    module.load_state_dict(new_sd, strict=False)
    print(f"[INFO] Quant load: matched={matched}, ignored_extra={ignored}, shape_mismatch={shape_mismatch}")
    
    if ignored_keys:
        print("\n[DEBUG] Ignored keys from checkpoint (not found in model):")
        for key in ignored_keys:
            print("  ", key)

    if mismatched_keys:
        print("\n[DEBUG] Shape mismatched keys:")
        for key, shape_model, shape_ck in mismatched_keys:
            print(f"  {key}: model={shape_model}, ckpt={shape_ck}")
            

def rotary_emb(rope_dim, seq_len, device, base=1_000_000.0, dtype=torch.bfloat16):
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,f->tf', t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def apply_rope(x, cos, sin, rope_dim):
    d = rope_dim
    x1 = x[..., :d]
    x2 = torch.stack([-x1[..., 1::2], x1[..., ::2]], dim=-1).reshape_as(x1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,d]
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rot = x1 * cos + x2 * sin
    return torch.cat([x_rot, x[..., d:]], dim=-1)


def _unpack_nibbles_interleaved(packed_u8: torch.Tensor, *, order="high_low") -> torch.Tensor:
    b = packed_u8.view(-1).to(torch.uint8)
    low  = b & 0x0F
    high = b >> 4
    out = torch.empty(b.numel() * 2, dtype=torch.uint8, device=b.device)
    if order == "low_high":
        out[0::2] = low;  out[1::2] = high
    else:
        out[0::2] = high; out[1::2] = low
    return out


def dequantize(qweight, scales, zeros, out_dim, in_dim, group_size=None, debug=False):
    dev = qweight.device
    total = out_dim * in_dim
    if qweight.numel() * 2 != total:
        raise ValueError(f"[ERROR] qweight size mismatch: {qweight.numel()} vs {total//2}")

    if scales.dim() == 2:
        out_s, n_groups = scales.shape
        assert out_s == out_dim
    else:
        n_groups = scales.numel() // out_dim
        scales = scales.view(out_dim, n_groups)
 
    if group_size is None:
        assert in_dim % n_groups == 0
        group_size = in_dim // n_groups

    codes = _unpack_nibbles_interleaved(qweight, order="high_low").to(torch.int16)
    codes[codes >= 8] -= 16  # symmetric: -8..7
    codes = codes.view(out_dim, in_dim)

    scales = scales.to(device=dev, dtype=torch.float32)
    out = torch.empty(out_dim, in_dim, dtype=torch.float32, device=dev)
    for g in range(n_groups):
        s = scales[:, g].unsqueeze(1)
        sl = slice(g * group_size, (g + 1) * group_size)
        out[:, sl] = codes[:, sl].to(torch.float32) * s
    return out


# -------------------- core layers --------------------

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, n_groups=12, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = n_groups
        self.group_size = in_features // n_groups
        self.register_buffer("qweight", torch.empty(out_features * in_features // 2, 1, dtype=torch.uint8))
        self.register_buffer("scales",  torch.empty(out_features, n_groups, dtype=torch.float32))
        self.register_buffer("zeros",   torch.empty(out_features * n_groups // 2, 1, dtype=torch.uint8))
        self.register_buffer("out_scale", torch.empty(out_features, dtype=torch.float32), persistent=False)  # for int8/fp8 output
        self.register_buffer("sparse_mask", torch.empty(0, dtype=torch.bool), persistent=False)  # optional 1D mask for sparse emulation
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        LPU-style execution with optional blocks:
          A/B buffers, unpack+scale (meta), sparse-select (optional),
          PE MAC accumulate into fp32 C buffer, PC(bias, ReLU, convert/quant), Data align.
        """
        if LPU_EMULATE:
            B, T, K = x.shape[0], x.shape[1], self.in_features
            N, G, gs = self.out_features, self.n_groups, self.group_size
            assert gs * G == K, f"group_size*groups must equal in_features ({gs}*{G}!={K})"

            # --- C Buffer (accumulator) & A view ---
            A = x.reshape(B*T, K).to(torch.float32)         # A Buffer reader emits fp32 for stable MAC
            C = x.new_zeros((B*T, N), dtype=torch.float32)  # C Buffer (wide)

            # --- Data shuffle / unpack / meta fetch (weights + scales) ---
            codes = _unpack_nibbles_interleaved(self.qweight.view(-1), order="high_low").to(torch.int16)
            codes[codes >= 8] -= 16
            codes = codes.view(N, K)                         # [N,K] int4 codes
            w_scales = self.scales.to(torch.float32)         # [N,G]

            # Align optional meta
            out_scale = self.out_scale if self.out_scale.numel() == N else None
            sparse_mask = self.sparse_mask if self.sparse_mask.numel() == G*gs else None
            if sparse_mask is not None:
                sparse_mask = sparse_mask.view(G, gs)        # [G, gs] bool

            # --- Stream on K (group tiles): LOAD → dequant → sparse-select → MAC ---
            for g in range(G):
                k0, k1 = g*gs, (g+1)*gs

                # A Buffer: K-slice
                a_tile = A[:, k0:k1]                         # [M, gs]

                # B Buffer: dequant current tile (unpack + multiply by per-group scale)
                # w_tile in [N, gs]
                w_tile = codes[:, k0:k1].to(torch.float32) * w_scales[:, g].unsqueeze(1)

                # Sparse select (optional) — matches "sparse_select" block
                mask_g = sparse_mask[g] if sparse_mask is not None else None
                a_tile, w_tile = _apply_sparse_select(a_tile, w_tile, mask_g)  # col-prune both sides

                # PE (MAC array): partial product accumulate into C Buffer
                # shapes: [M, k_sel] @ [k_sel, N] → [M, N]
                if a_tile.numel() > 0:
                    C += a_tile @ w_tile.t()

            # --- PC (post process): bias, ReLU (optional), convert/quant, data align ---
            y = _pc_postprocess(C,
                                self.bias,
                                LPU_RELU,
                                LPU_OUT_DTYPE,
                                out_scale,
                                ref_dtype=x.dtype)

            # Data align / writer: reshape back to [B,T,N] (no-op layout change)
            return y.view(B, T, N)

        # ---- Fallback: original dense path (kept for A/B checks) ----
        W = dequantize(self.qweight, self.scales, self.zeros,
                       self.out_features, self.in_features,
                       group_size=self.group_size)
        out32 = x.to(torch.float32) @ W.t()
        if self.bias is not None:
            out32 = out32 + self.bias
        return out32.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.eps = eps
    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)     # RS
        x_hat = x * torch.rsqrt(var + self.eps).to(x.dtype)         # PVPU
        return (x_hat * self.weight).to(x.dtype)


# -------------------- config wrapper --------------------

class Config:
    def __init__(self, hf_cfg):
        self.vocab_size        = hf_cfg.vocab_size
        self.hidden_size       = hf_cfg.hidden_size
        self.n_head            = hf_cfg.num_attention_heads
        self.n_kv_head         = hf_cfg.num_key_value_heads
        self.intermediate_size = hf_cfg.intermediate_size
        self.rope_theta        = getattr(hf_cfg, "rope_theta", 1_000_000.0)


# -------------------- attention / mlp / blocks --------------------

class RotaryEmbedding(nn.Module):
    """HF-compatible rotary embedding that returns per-batch cos/sin."""
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        # position_ids: [B, T]
        t = position_ids.to(self.inv_freq.device).float()        # [B,T]
        freqs = torch.einsum("bt,f->btf", t, self.inv_freq)      # [B,T,D/2]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)      # [B,T,D]
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        return cos.to(x.dtype), sin.to(x.dtype)


class QwenAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # Robust config access
        self.n_head = getattr(config, "n_head", getattr(config, "num_attention_heads", None))
        self.n_kv_head = getattr(config, "n_kv_head", getattr(config, "num_key_value_heads", None))
        self.hidden_size = getattr(config, "hidden_size")
        self.rope_theta  = getattr(config, "rope_theta", 1_000_000.0)
        if self.hidden_size is None or self.n_head is None or self.n_kv_head is None:
            raise ValueError(
                f"Missing one of required attributes: hidden_size={self.hidden_size}, "
                f"n_head={self.n_head}, n_kv_head={self.n_kv_head}"
            )

        self.head_dim = self.hidden_size // self.n_head

        self.q_proj = QuantLinear(self.hidden_size, self.n_head    * self.head_dim, n_groups=12, bias=True)
        self.k_proj = QuantLinear(self.hidden_size, self.n_kv_head * self.head_dim, n_groups=12, bias=True)
        self.v_proj = QuantLinear(self.hidden_size, self.n_kv_head * self.head_dim, n_groups=12, bias=True)
        self.o_proj = QuantLinear(self.n_head * self.head_dim,     self.hidden_size, n_groups=12, bias=False)

        # HF-style rotary
        self.rotary_emb = RotaryEmbedding(self.head_dim, self.rope_theta)

    def forward(self, x):
        B, T, C = x.shape
        H, HK = self.n_head, self.n_kv_head
        Dh_q = self.q_proj.out_features // H
        Dkv  = self.k_proj.out_features // HK
        assert Dh_q == self.head_dim and Dkv == self.head_dim, \
            f"head_dim mismatch: {Dh_q} vs {Dkv} vs {self.head_dim}"

        # projections (LPU-style MM is inside QuantLinear)
        q = self.q_proj(x).view(B, T, H,  Dh_q).transpose(1, 2)   # [B,H,T,D]
        k = self.k_proj(x).view(B, T, HK, Dkv ).transpose(1, 2)   # [B,HK,T,D]
        v = self.v_proj(x).view(B, T, HK, Dkv ).transpose(1, 2)   # [B,HK,T,D]

        # RoPE (PVPU-style elementwise)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B,T]
        cos, sin = self.rotary_emb(x, position_ids)                                # [B,T,D]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                # [B,*,T,D]

        # GQA expand K/V to full heads
        rep = H // HK
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)  # [B,H,T,D]
            v = v.repeat_interleave(rep, dim=1)  # [B,H,T,D]

        # ----- scaled dot-product attention (fp32, RS+PVPU decomposition) -----
        scale = self.head_dim ** -0.5
        qf = q.float(); kf = k.float(); vf = v.float()

        # scores: [B,H,T,T]  (MM on LPU in reality; here torch.matmul)
        scores = torch.matmul(qf, kf.transpose(2, 3)) * scale

        # causal mask (PVPU add)
        mask = torch.triu(
            torch.full((1, 1, T, T), float("-inf"), device=x.device, dtype=torch.float32),
            diagonal=1
        )
        scores = scores + mask

        # stable softmax (RS max, PVPU exp, RS sum, PVPU div)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        scores = scores.clamp(-50, 50)
        probs = torch.softmax(scores, dim=-1)

        # context: [B,H,T,D]  (MM on LPU; here torch.matmul)
        y = torch.matmul(probs, vf)
        # ---------------------------------------------------------------

        y = y.transpose(1, 2).reshape(B, T, H * Dh_q)            # [B,T,C]
        return self.o_proj(y.to(x.dtype))                        # bf16 out


class Mlp(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.gate_proj = QuantLinear(config.hidden_size,       config.intermediate_size, n_groups=12, bias=False)
        self.up_proj   = QuantLinear(config.hidden_size,       config.intermediate_size, n_groups=12, bias=False)
        self.down_proj = QuantLinear(config.intermediate_size, config.hidden_size,       n_groups=70, bias=False)
    def forward(self, x):
        g = self.gate_proj(x).float()
        u = self.up_proj(x).float()
        h = F.silu(g) * u                  # PVPU-style elementwise
        return self.down_proj(h).to(x.dtype)  # keep bf16 flow


class DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = QwenAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = Mlp(config, layer_idx)
    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x) + residual
        residual = x
        x = self.post_attention_layernorm(x)
        return self.mlp(x) + residual


class QwenModel(nn.Module):
    def __init__(self, config: Config, n_layer: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens.weight.data = self.embed_tokens.weight.data.to(torch.bfloat16)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(n_layer)])
        self.norm = RMSNorm(config.hidden_size)
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids).to(torch.bfloat16)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# -------------------- top module --------------------

class QwenVlaQuant(nn.Module):
    def __init__(self, ckpt_path: str, hf_base: str):
        super().__init__()
        print("[INFO] Loading HF config/weights from:", hf_base)
        hf = AutoModelForCausalLM.from_pretrained(hf_base, trust_remote_code=True)  # stays on CPU
        cfg = Config(hf.config)

        self.n_layer = hf.config.num_hidden_layers
        self.model = QwenModel(cfg, self.n_layer)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # tie lm head to embeddings
        self.lm_head.weight = self.model.embed_tokens.weight

        if ckpt_path:
            print("[INFO] Auto-loading quantized weights from:", ckpt_path)
            load_quant_state_strict_subset(self, ckpt_path, device="cpu")

        # copy HF embeddings (AWQ ckpts usually don't store them)
        with torch.no_grad():
            self.model.embed_tokens.weight.data.copy_(hf.model.embed_tokens.weight.data.to(torch.bfloat16))
            self.lm_head.weight = self.model.embed_tokens.weight  # ensure tie

        # copy HF biases for q/k/v (AWQ weight-only often lacks bias)
        with torch.no_grad():
            for i in range(self.n_layer):
                mq  = self.model.layers[i].self_attn
                hfa = hf.model.layers[i].self_attn
                for name in ["q_proj", "k_proj", "v_proj"]:
                    ql = getattr(mq, name)
                    if ql.bias is not None:
                        ql.bias.copy_(getattr(hfa, name).bias.to(ql.bias.dtype))
                        
        # copy HF RMSNorm gammas (input/post-attn per layer + final norm)
        with torch.no_grad():
            for i in range(self.n_layer):
                my_layer = self.model.layers[i]
                hf_layer = hf.model.layers[i]

                my_layer.input_layernorm.weight.copy_(
                    hf_layer.input_layernorm.weight.to(my_layer.input_layernorm.weight.dtype)
                )
                my_layer.post_attention_layernorm.weight.copy_(
                    hf_layer.post_attention_layernorm.weight.to(my_layer.post_attention_layernorm.weight.dtype)
                )

            self.model.norm.weight.copy_(
                hf.model.norm.weight.to(self.model.norm.weight.dtype)
            )
            
        # Use HF rotary for all layers (matches HF math exactly)
        with torch.no_grad():
            for i in range(self.n_layer):
                self.model.layers[i].self_attn.rotary_emb = hf.model.rotary_emb


    def forward(self, input_ids, collect_debug=False):
        hidden = self.model(input_ids)
        hidden = hidden.to(self.lm_head.weight.dtype)
        return self.lm_head(hidden)


# -------------------- entry --------------------

if __name__ == "__main__":
    # Defaults (override with env vars if needed)
    CKPT    = os.environ.get("QWEN_CKPT",   "/home/yueyuhao/quantised_result/qwen2-1_5b-awq-w4g128/model.safetensors")
    HF      = os.environ.get("QWEN_HF",     "/home/yueyuhao/.cache/modelscope/hub/models/Qwen/Qwen2-1.5B")
    DEVICE  = os.environ.get("QWEN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = int(os.environ.get("QWEN_SEQLEN", "32"))

    torch.manual_seed(0)
    device = torch.device(DEVICE)

    model = QwenVlaQuant(CKPT, HF).to(device).eval()

    vocab_size = model.lm_head.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), device=device)

    with torch.no_grad():
        out = model(input_ids)

    print("✅ Output shape:", tuple(out.shape))
    print("Mean:", out.mean().item(), "Std:", out.std().item())

    # (Optional) quick A/B check vs fallback path
    if bool(int(os.environ.get("CHECK_AB", "0"))):
        os.environ["LPU_EMULATE"] = "0"
        with torch.no_grad():
            ref = model(input_ids)
        os.environ["LPU_EMULATE"] = "1"
        with torch.no_grad():
            lpu = model(input_ids)
        diff = (ref - lpu).abs()
        print("A/B max abs diff:", diff.max().item(), "mean abs diff:", diff.mean().item())
