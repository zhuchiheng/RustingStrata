#!/usr/bin/env python3
"""Convert Inkling-2-Mini's original mtp.safetensors (BF16, HF names) into the GGUF that
Strata's inkling2/dspark.rs reads.

Strata expects architecture "dflash" with these names (see src/inkling2/dspark.rs):
  dflash.block_count / embedding_length / feed_forward_length / attention.head_count /
  attention.head_count_kv / attention.key_length / attention.layer_norm_rms_epsilon /
  rope.freq_base / block_size / attention.sliding_window /
  attention.sliding_window_pattern / target_layers
  dspark_fuse.weight, dspark_norm.weight, mask_embd.weight, output_norm.weight,
  markov_w1.weight, joint_proj.weight/.bias, dspark_hidden_out.weight/.bias,
  blk.N.{dflare_norm,attn_k_ctx,attn_v_ctx,attn_k_ctx_norm,attn_norm,attn_q,attn_k,attn_v,
        attn_output,attn_q_norm,attn_k_norm,ffn_norm,ffn_gate,ffn_up,ffn_down}

The source (mtp.safetensors) names them:
  fuse_weight, norm.weight, mask_embed, markov_head.markov_w1.weight,
  markov_head.rnn_proj.weight/.bias, markov_head.hidden_out.weight/.bias,
  markov_head.dflare_norm.weight,
  layers.N.{dflare_norm,input_layernorm,post_attention_layernorm,
            self_attn.{q,k,v,o}_proj,self_attn.{q,k}_norm,self_attn.{k,v}_proj_ctx,
            mlp.{gate,up,down}_proj}

Everything is written as F32 (the drafter is small, 0.6 GiB in BF16) so no quant type has to be
matched at load; Strata's Mat reads F32 fine.
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
T_UINT32, T_STRING, T_BOOL, T_FLOAT32, T_ARRAY = 4, 8, 7, 6, 9
# GGUF tensor types
GGML_F32 = 0
GGUF_ALIGN = 32

LAYER_MAP = {
    "dflare_norm": "dflare_norm.weight",
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
    "attn_q": "self_attn.q_proj.weight",
    "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight",
    "attn_k_norm": "self_attn.k_norm.weight",
    "attn_k_ctx": "self_attn.k_proj_ctx.weight",
    "attn_v_ctx": "self_attn.v_proj_ctx.weight",
    "attn_k_ctx_norm": "self_attn.k_norm_ctx.weight",
    "ffn_gate": "mlp.gate_proj.weight",
    "ffn_up": "mlp.up_proj.weight",
    "ffn_down": "mlp.down_proj.weight",
}


def read_safetensors(path):
    """Return {name: (shape, np.float32 array)} plus the raw header."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            beg, end = info["data_offsets"]
            f.seek(base + beg)
            raw = f.read(end - beg)
            dt = info["dtype"]
            shape = tuple(info["shape"])
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                arr = f32.reshape(shape)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            else:
                raise SystemExit(f"{name}: unsupported dtype {dt}")
            out[name] = (shape, np.ascontiguousarray(arr, dtype=np.float32))
        return out


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_uint32(k, v):
    return gguf_str(k) + struct.pack("<I", T_UINT32) + struct.pack("<I", v)


def kv_float32(k, v):
    return gguf_str(k) + struct.pack("<I", T_FLOAT32) + struct.pack("<f", v)


def kv_string(k, v):
    return gguf_str(k) + struct.pack("<I", T_STRING) + gguf_str(v)


def kv_bool_array(k, vals):
    out = gguf_str(k) + struct.pack("<I", T_ARRAY) + struct.pack("<I", T_BOOL) + struct.pack("<Q", len(vals))
    return out + b"".join(struct.pack("<B", 1 if v else 0) for v in vals)


def kv_uint32_array(k, vals):
    out = gguf_str(k) + struct.pack("<I", T_ARRAY) + struct.pack("<I", T_UINT32) + struct.pack("<Q", len(vals))
    return out + b"".join(struct.pack("<I", v) for v in vals)


GGML_Q8_0 = 8


def q8_0_row(row):
    """One row of f32 as GGML Q8_0: blocks of 32 values, an f16 scale then 32 int8 (34 bytes a block).

    d = amax / 127 and the codes round to nearest, which is what the dequantizer in src/quant/mod.rs reads back.
    A row whose length is not a multiple of 32 cannot be written this way; the caller keeps it F32.
    """
    import numpy as np
    v = np.asarray(row, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(v).max(axis=1)
    d = (amax / 127.0).astype(np.float16)
    inv = np.where(amax > 0, 127.0 / np.where(amax > 0, amax, 1.0), 0.0).astype(np.float32)
    q = np.clip(np.rint(v * inv[:, None]), -127, 127).astype(np.int8)
    out = bytearray()
    for b in range(v.shape[0]):
        out += d[b].tobytes()
        out += q[b].tobytes()
    return bytes(out)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "mtp.safetensors")
    cfg_path = Path(sys.argv[2] if len(sys.argv) > 2 else "config.json")
    dst = Path(sys.argv[3] if len(sys.argv) > 3 else "dspark-Inkling-2-Mini-F32.gguf")

    cfg = json.loads(cfg_path.read_text())["mtp_config"]
    n_layer = cfg["num_hidden_layers"]
    n_embd = cfg["hidden_size"]
    n_ff = cfg["intermediate_size"]
    n_head = cfg["num_attention_heads"]
    n_head_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    rope = cfg["rope_theta"]
    block_size = cfg["block_size"]
    n_swa = cfg["sliding_window_size"]
    local = set(cfg["local_layer_ids"])
    targets = cfg["target_layer_ids"]

    t = read_safetensors(src)
    print(f"source tensors: {len(t)}")

    # Build the output tensor list under Strata's names.
    out = {}

    def put(dst_name, src_name, shape_hint=None):
        if src_name not in t:
            raise SystemExit(f"missing source tensor {src_name}")
        shape, arr = t[src_name]
        out[dst_name] = arr
        print(f"  {src_name:48} -> {dst_name:28} {shape}")

    put("dspark_fuse.weight", "fuse_weight")
    put("dspark_norm.weight", "norm.weight")
    put("mask_embd.weight", "mask_embed")
    put("output_norm.weight", "norm.weight")  # Strata reads output_norm too; same row norm
    put("markov_w1.weight", "markov_head.markov_w1.weight")
    put("joint_proj.weight", "markov_head.rnn_proj.weight")
    put("joint_proj.bias", "markov_head.rnn_proj.bias")
    put("dspark_hidden_out.weight", "markov_head.hidden_out.weight")
    put("dspark_hidden_out.bias", "markov_head.hidden_out.bias")

    for il in range(n_layer):
        for dst_suffix, src_suffix in LAYER_MAP.items():
            put(f"blk.{il}.{dst_suffix}.weight", f"layers.{il}.{src_suffix}")

    # GGUF tensor order: name, n_dims, dims (little endian u64), type, offset.
    names = sorted(out)
    header = b""
    header += kv_string("general.architecture", "dflash")
    header += kv_string("general.name", "dspark-Inkling-2-Mini")
    header += kv_uint32("dflash.block_count", n_layer)
    header += kv_uint32("dflash.embedding_length", n_embd)
    header += kv_uint32("dflash.feed_forward_length", n_ff)
    header += kv_uint32("dflash.attention.head_count", n_head)
    header += kv_uint32("dflash.attention.head_count_kv", n_head_kv)
    header += kv_uint32("dflash.attention.key_length", head_dim)
    header += kv_float32("dflash.attention.layer_norm_rms_epsilon", eps)
    header += kv_float32("dflash.rope.freq_base", rope)
    header += kv_uint32("dflash.block_size", block_size)
    header += kv_uint32("dflash.attention.sliding_window", n_swa)
    header += kv_bool_array("dflash.attention.sliding_window_pattern", [i in local for i in range(n_layer)])
    header += kv_uint32_array("dflash.target_layers", targets)
    header += kv_uint32("general.alignment", GGUF_ALIGN)
    kv_count = 15

    info = b""
    offset = 0
    q8 = os.environ.get("STRATA_DSPARK_Q8", "") not in ("", "0")
    blobs = []
    # Tensors that share a source array (output_norm == dspark_norm) are written once and their
    # offsets point at the same bytes; GGUF allows that and it keeps the offsets consistent.
    seen = {}
    for name in names:
        arr = out[name]
        dims = list(reversed(arr.shape))
        key = (arr.__array_interface__["data"][0], arr.shape, arr.nbytes)
        if key in seen:
            at = seen[key]
        else:
            at = offset
            seen[key] = at
            b = arr.tobytes()
            ty = GGML_F32
            if q8 and arr.ndim == 2 and arr.shape[1] % 32 == 0:
                b = b"".join(q8_0_row(r) for r in arr)
                ty = GGML_Q8_0
            blobs.append(b)
            offset += len(b)
        info += gguf_str(name)
        info += struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", d)
        info += struct.pack("<I", ty)
        info += struct.pack("<Q", at)

    head = GGUF_MAGIC + struct.pack("<I", GGUF_VERSION) + struct.pack("<Q", len(names)) + struct.pack("<Q", kv_count)
    # The tensor data must start on a general.alignment boundary: Strata rounds the offset up
    # (gguf.rs: data_offset = next_multiple_of(align)), so the padding has to be in the file.
    body = head + header + info
    pad = (-len(body)) % GGUF_ALIGN
    with open(dst, "wb") as f:
        f.write(body)
        f.write(bytes(pad))
        for b in blobs:
            f.write(b)
    print(f"header+index {len(body)} bytes, padding {pad} to {GGUF_ALIGN}")
    print(f"\nwrote {dst} ({dst.stat().st_size / 1048576:.1f} MiB, {len(names)} tensors)")


if __name__ == "__main__":
    main()
