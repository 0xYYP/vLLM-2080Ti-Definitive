"""MTP verify causal 校验：kernel（causal mask 修复后）vs right-aligned causal 参考。

seq_len_cpu 含 4 个 speculative token；query 行 i（0..3）位置 = kv_len-4+i，
attend 边界 kv_idx < kv_len-4+1+i。随机 K/V scale + 随机 page table + 尾页。
"""
import sys, torch
sys.path.insert(0, "/opt/vllm-2080ti-definitive")
from vllm.v1.attention.backends.triton_attn import _int8kv_decode_jit_args
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

head_dim, num_heads, num_kv_heads, page_size, gqa = 256, 12, 2, 1504, 6
sm_scale = 0.0625
qo_len = 4
torch.cuda.set_device(0)


def run_kernel(q, kv, k_scale, v_scale, seq_len, indices):
    nblocks = (seq_len + page_size - 1) // page_size
    lpl = seq_len - (nblocks - 1) * page_size
    k_view, v_view = kv[:, 0][..., :head_dim], kv[:, 1][..., :head_dim]
    out = torch.empty(4, num_heads, head_dim, device="cuda", dtype=torch.float16)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    w = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", jit_args=_int8kv_decode_jit_args(head_dim))
    indptr = torch.tensor([0, nblocks], dtype=torch.int32, device="cuda")
    lpl_t = torch.tensor([lpl], dtype=torch.int32, device="cuda")
    w.plan(indptr, indices, lpl_t, num_heads, num_kv_heads, head_dim, page_size,
           q_data_type=q.dtype, kv_data_type=kv.dtype, o_data_type=out.dtype,
           sm_scale=sm_scale, q_len_per_req=qo_len)
    w.run(q, (k_view, v_view), k_scale, v_scale, 0.0, sm_scale, 1.0, 1e-4, gqa,
          k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), out=out)
    torch.cuda.synchronize()
    return out


def ref_impl(q, kv, k_scale, v_scale, seq_len, causal, indices):
    kvh = torch.arange(num_heads) // gqa
    # 按物理页（indices）顺序读 KV 与 scale
    idx = [i.item() for i in indices]
    k_flat = torch.cat([kv[i][0][..., :head_dim] for i in idx]).float().reshape(-1, num_kv_heads, head_dim)[:seq_len]
    v_flat = torch.cat([kv[i][1][..., :head_dim] for i in idx]).float().reshape(-1, num_kv_heads, head_dim)[:seq_len]
    ks_flat = torch.cat([k_scale[i] for i in idx]).float().reshape(-1, num_kv_heads)[:seq_len]
    vs_flat = torch.cat([v_scale[i] for i in idx]).float().reshape(-1, num_kv_heads)[:seq_len]
    qf = q.float()
    o = torch.zeros(4, num_heads, head_dim, device="cuda")
    for b in range(4):
        for h in range(num_heads):
            kv = kvh[h].item()
            logits = (qf[b, h] @ k_flat[:, kv, :].T) * ks_flat[:, kv] * sm_scale
            if causal:
                # query row b sits at position seq_len - qo_len + b
                bound = seq_len - qo_len + 1 + b
                logits[bound:] = float("-inf")
            p = torch.softmax(logits.float(), dim=-1)
            o[b, h] = p @ (v_flat[:, kv, :] * vs_flat[:, kv].unsqueeze(-1))
    return o


def make_case(seq_len, n_kv_pages):
    nblocks = (seq_len + page_size - 1) // page_size
    kv = torch.randint(-50, 50, (nblocks, 2, page_size, num_kv_heads, 272),
                       dtype=torch.int8, device="cuda")
    base = kv.view(torch.int32)
    sb, ss, sh = kv.stride(0) // 4, kv.stride(2) // 4, kv.stride(3) // 4
    k_scale = torch.as_strided(base, size=(nblocks, page_size, num_kv_heads),
                               stride=(sb, ss, sh), storage_offset=head_dim // 4).float()
    v_scale = torch.as_strided(base, size=(nblocks, page_size, num_kv_heads),
                               stride=(sb, ss, sh),
                               storage_offset=head_dim // 4 + (kv.stride(1) // 4)).float()
    k_scale.uniform_(0.001, 0.2)
    v_scale.uniform_(0.001, 0.2)
    # 随机 page table：物理页号随机（kv 分配 nblocks 页，indices 用 0..nblocks-1 的随机排列）
    idx = torch.randperm(nblocks, dtype=torch.int32, device="cuda")
    return kv, k_scale, v_scale, idx


def main():
    # 每个 seq 都放大最后 qo_len-1 个 speculative token 的 k_scale（logits 主导强判别）
    for seq_len in (125000, 4096, 2000, 500):
        kv, k_scale, v_scale, idx = make_case(seq_len, seq_len // page_size + 1)
        amp_t = torch.arange(seq_len - qo_len + 1, seq_len, device="cuda")
        phys_pg = idx[amp_t // page_size]  # request 内页 → 物理页
        k_scale[phys_pg, amp_t % page_size, :] = 10.0
        q = torch.randn(4, num_heads, head_dim, device="cuda", dtype=torch.float16)
        out = run_kernel(q, kv, k_scale, v_scale, seq_len, idx)
        ref_causal = ref_impl(q, kv, k_scale, v_scale, seq_len, causal=True, indices=idx)
        ref_full = ref_impl(q, kv, k_scale, v_scale, seq_len, causal=False, indices=idx)
        d_c = (out.float() - ref_causal).abs()
        d_f = (out.float() - ref_full).abs()
        r_c = d_c.sum() / ref_causal.float().abs().sum()
        r_f = d_f.sum() / ref_full.float().abs().sum()
        print(f"[seq={seq_len}] AMP causal ref: rel_l1={r_c.item():.6f} | full ref: rel_l1={r_f.item():.6f}")
        if seq_len == 125000:
            hd = d_c.mean(-1).mean(0)  # (4,12)
            print("  causal diff per row (query 0-3):", " ".join(f"{hd[i].mean():.6f}" for i in range(4)))
    print("RESULT: 每个 seq 的 causal ref 应显著小于 full ref（causal mask 契约正确）")


if __name__ == "__main__":
    main()
