import torch

# standard attention shape
sequence = 8
headdim = 4

# init fwd input
q = torch.rand(sequence, headdim, dtype=float)
k = torch.rand_like(q, dtype=float)
v = torch.rand_like(q, dtype=float)

# standard fwd attention
kt = torch.transpose(k, 0, 1)
s = torch.mm(q, kt)
max, _ = torch.max(s, dim=1)
s_sub = s - max.reshape(-1, 1)
s_exp = torch.exp(s_sub)
sum = torch.sum(s_exp, dim=1)
p = s_exp / sum.reshape(-1, 1)
p_standard = torch.softmax(s, 1, dtype=float)

l = max + torch.log(sum)
o = torch.mm(p, v)

# init bwd input
do = torch.rand_like(o, dtype=float)

# standard bwd attention
pt = torch.transpose(p, 0, 1)
dv = torch.mm(pt, do)
vt = torch.transpose(v, 0, 1)
dp = torch.mm(do, vt)
ds = torch.zeros_like(dp, dtype=float)

for i in range(sequence):
    # 切片矩阵的一行，高维度消失，进行矩阵乘运算需要reshape
    di = torch.mm(p[i, :].reshape(1, -1), dp[i, :].reshape(1, -1).transpose(-2, -1))
    scale = dp[i, :] - di
    ds[i:] = torch.mul(scale, p[i, :])

dq = torch.mm(ds, k)
dst = torch.transpose(ds, 0, 1)
dk = torch.mm(dst, q)

# flash attention shape
blkq = 2
blkk = 2

# flash attention bwd init output
fa_dq = torch.zeros_like(dq, dtype=float)
fa_dk = torch.zeros_like(dk, dtype=float)
fa_dv = torch.zeros_like(dv, dtype=float)
fa_D = torch.mul(o, do).sum(dim=1)

for it_k in range(sequence // blkk):
    l_k = k[it_k * blkk : (it_k + 1) * blkk, :]
    l_v = v[it_k * blkk : (it_k + 1) * blkk, :]
    l_dk = torch.zeros_like(l_k, dtype=float)
    l_dv = torch.zeros_like(l_v, dtype=float)
    for it_q in range(sequence // blkq):
        l_q = q[it_q * blkq : (it_q + 1) * blkq, :]
        l_do = do[it_q * blkq : (it_q + 1) * blkq, :]
        l_s = torch.mm(l_q, l_k.transpose(-2, -1))
        l_p = torch.exp(l_s - l[it_q * blkq : (it_q + 1) * blkq].reshape(-1, 1))
        l_dv = l_dv + torch.mm(l_p.transpose(-2, -1), l_do)
        l_dp = torch.mm(l_do, l_v.transpose(-2, -1))
        l_scale = l_dp - fa_D[it_q * blkq : (it_q + 1) * blkq].reshape(-1, 1)
        l_ds = torch.mul(l_p, l_scale)
        l_dq = torch.mm(l_ds, l_k)
        fa_dq[it_q * blkq : (it_q + 1) * blkq, :] = (
            fa_dq[it_q * blkq : (it_q + 1) * blkq, :] + l_dq
        )
        l_dk = l_dk + torch.mm(l_ds.transpose(-2, -1), l_q)
    fa_dk[it_k * blkk : (it_k + 1) * blkk, :] = l_dk
    fa_dv[it_k * blkk : (it_k + 1) * blkk, :] = l_dv

# check fa and ref
print(f"dv max diff: {(fa_dv - dv).abs().max().item()}")  # pass
print(f"dk max diff: {(fa_dk - dk).abs().max().item()}")  # pass
print(f"dq max diff: {(fa_dq - dq).abs().max().item()}")  # pass
