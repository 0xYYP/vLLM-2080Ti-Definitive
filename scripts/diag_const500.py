import sys, torch
sys.path.insert(0, "/opt/vllm-2080ti-definitive")
from vllm.v1.attention.backends.triton_attn import _int8kv_decode_jit_args
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

head_dim, num_heads, num_kv_heads, page_size, gqa = 256, 12, 2, 1504, 6
sm_scale = 0.0625
torch.cuda.set_device(0)
seq_len = 500

kv = torch.randint(-50, 50, (1, 2, page_size, num_kv_heads, 272), dtype=torch.int8, device="cuda")
base = kv.view(torch.int32)
sb, ss, sh = kv.stride(0) // 4, kv.stride(2) // 4, kv.stride(3) // 4
k_scale = torch.as_strided(base, size=(1, page_size, num_kv_heads),
                           stride=(sb, ss, sh), storage_offset=head_dim // 4).float()
v_scale = torch.as_strided(base, size=(1, page_size, num_kv_heads),
                           stride=(sb, ss, sh),
                           storage_offset=head_dim // 4 + (kv.stride(1) // 4)).float()
k_scale.fill_(0.05)
v_scale.fill_(0.02)

q = torch.randn(4, num_heads, head_dim, device="cuda", dtype=torch.float16)
k_view, v_view = kv[:, 0][..., :head_dim], kv[:, 1][..., :head_dim]
out = torch.empty(4, num_heads, head_dim, device="cuda", dtype=torch.float16)
ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
w = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", jit_args=_int8kv_decode_jit_args(head_dim))
indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
indices = torch.tensor([0], dtype=torch.int32, device="cuda")
lpl_t = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
w.plan(indptr, indices, lpl_t, num_heads, num_kv_heads, head_dim, page_size,
       q_data_type=q.dtype, kv_data_type=kv.dtype, o_data_type=out.dtype,
       sm_scale=sm_scale, q_len_per_req=4)
w.run(q, (k_view, v_view), k_scale, v_scale, 0.0, sm_scale, 1.0, 1e-4, gqa,
      k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), out=out)
torch.cuda.synchronize()

# ref
kvh = torch.arange(num_heads) // gqa
k_flat = kv[:, 0][..., :head_dim].float().reshape(-1, num_kv_heads, head_dim)[:seq_len]
v_flat = kv[:, 1][..., :head_dim].float().reshape(-1, num_kv_heads, head_dim)[:seq_len]
ks = k_flat.new_full((seq_len, num_kv_heads), 0.05)
vs = v_flat.new_full((seq_len, num_kv_heads), 0.02)
ref = torch.zeros(4, num_heads, head_dim, device="cuda")
qf = q.float()
for b in range(4):
    for h in range(num_heads):
        kvh_ = kvh[h].item()
        logits = (qf[b, h] @ k_flat[:, kvh_, :].T) * ks[:, kvh_]
        p = torch.softmax(logits.float(), dim=-1)
        ref[b, h] = p @ (v_flat[:, kvh_, :] * vs[:, kvh_].unsqueeze(-1))

diff = (out.float() - ref).abs()
rel = diff.sum() / ref.float().abs().sum()
print(f"seq={seq_len} max_abs={diff.max().item():.6f} mean_abs={diff.mean().item():.6f} rel_l1={rel.item():.6f}")
for b in range(2):
    for h in range(2):
        print(f"  out[{b},{h},:6] =", [f"{x:.4f}" for x in out[b, h, :6].float().tolist()])
        print(f"  ref[{b},{h},:6] =", [f"{x:.4f}" for x in ref[b, h, :6].tolist()])
print("  out[0,0,120:124] =", [f"{x:.4f}" for x in out[0, 0, 120:124].float().tolist()])
print("  ref[0,0,120:124] =", [f"{x:.4f}" for x in ref[0, 0, 120:124].tolist()])
print("  out[0,0,240:244] =", [f"{x:.4f}" for x in out[0, 0, 240:244].float().tolist()])
print("  ref[0,0,240:244] =", [f"{x:.4f}" for x in ref[0, 0, 240:244].tolist()])
