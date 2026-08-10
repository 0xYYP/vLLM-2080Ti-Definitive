import sys

import torch

sys.path.insert(0, "/opt/vllm-2080ti-definitive")
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from vllm.v1.attention.backends.triton_attn import _int8kv_decode_jit_args

head_dim, num_heads, num_kv_heads, page_size, gqa = 256, 12, 2, 1504, 6
sm_scale = 0.0625
torch.cuda.set_device(0)


def run_kernel(q, kv, k_scale, v_scale, seq_len):
    nblocks = (seq_len + page_size - 1) // page_size
    lpl = seq_len - (nblocks - 1) * page_size
    k_view, v_view = kv[:, 0][..., :head_dim], kv[:, 1][..., :head_dim]
    out = torch.empty(4, num_heads, head_dim, device="cuda", dtype=torch.float16)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    w = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", jit_args=_int8kv_decode_jit_args(head_dim))
    indptr = torch.tensor([0, nblocks], dtype=torch.int32, device="cuda")
    indices = torch.arange(0, nblocks, dtype=torch.int32, device="cuda")
    lpl_t = torch.tensor([lpl], dtype=torch.int32, device="cuda")
    w.plan(indptr, indices, lpl_t, num_heads, num_kv_heads, head_dim, page_size,
           q_data_type=q.dtype, kv_data_type=kv.dtype, o_data_type=out.dtype,
           sm_scale=sm_scale, q_len_per_req=4)
    w.run(q, (k_view, v_view), k_scale, v_scale, 0.0, sm_scale, 1.0, 1e-4, gqa,
          k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), out=out)
    torch.cuda.synchronize()
    return out


def ref_impl(q, kv, k_scale, v_scale, seq_len):
    """per-token-head scale attention 参考实现（无 causal，全部 attend）。"""
    kvh = torch.arange(num_heads) // gqa  # (12,)
    k_flat = kv[:, 0][..., :head_dim].float().reshape(-1, num_kv_heads, head_dim)[:seq_len]  # (seq,2,256)
    v_flat = kv[:, 1][..., :head_dim].float().reshape(-1, num_kv_heads, head_dim)[:seq_len]
    ks_flat = k_scale.float().reshape(-1, num_kv_heads)[:seq_len]
    vs_flat = v_scale.float().reshape(-1, num_kv_heads)[:seq_len]
    qf = q.float()  # (4,12,256)
    o = torch.zeros(4, num_heads, head_dim)
    for b in range(4):
        for h in range(num_heads):
            kv = kvh[h].item()
            logits = (qf[b, h] @ k_flat[:, kv, :].T) * ks_flat[:, kv] * 0.0625  # sm_scale
            p = torch.softmax(logits.float(), dim=-1)
            o[b, h] = p @ (v_flat[:, kv, :] * vs_flat[:, kv].unsqueeze(-1))
    return o


def make_scales(nblocks):
    """随机 per-token per-head scale，写入 int8 存储的 padding 区（模拟服务布局）。"""
    kv = torch.randint(-50, 50, (nblocks, 2, page_size, num_kv_heads, 272),
                       dtype=torch.int8, device="cuda")
    base = kv.view(torch.int32)
    sb, ss, sh = kv.stride(0) // 4, kv.stride(2) // 4, kv.stride(3) // 4
    scale_off = head_dim // 4
    k_scale = torch.as_strided(base, size=(nblocks, page_size, num_kv_heads),
                               stride=(sb, ss, sh), storage_offset=scale_off).float()
    v_scale = torch.as_strided(base, size=(nblocks, page_size, num_kv_heads),
                               stride=(sb, ss, sh),
                               storage_offset=scale_off + (kv.stride(1) // 4)).float()
    k_scale.uniform_(0.001, 0.2)
    v_scale.uniform_(0.001, 0.2)
    return kv, k_scale, v_scale


def main():
    results = {}
    for seq_len, label in [(4096, "partition-4K"), (500, "non-partition-500")]:
        nblocks = (seq_len + page_size - 1) // page_size
        kv, k_scale, v_scale = make_scales(nblocks)
        q = torch.randn(4, num_heads, head_dim, device="cuda", dtype=torch.float16)
        out = run_kernel(q, kv, k_scale, v_scale, seq_len)
        ref = ref_impl(q, kv, k_scale, v_scale, seq_len)
        diff = (out.float() - ref).abs()
        rel = diff.sum() / ref.float().abs().sum()
        results[label] = (diff.max().item(), diff.mean().item(), rel.item())
        print(f"[{label}] max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f} rel_l1={rel.item():.6f}")
    ok = all(r[2] < 0.02 for r in results.values())
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
