import torch
import torch.nn.functional as F

batch = 1
seq = 8
headdim = 4
tileQ = 2
tileK = 2
# 整除运算符//
chunkQ = seq//tileQ
chunkK = seq//tileK

q = torch.rand(seq,headdim,dtype=torch.float32)
k = torch.rand(seq,headdim,dtype=torch.float32)
v = torch.rand(seq,headdim,dtype=torch.float32)

#TODO: standard self-attention
s_ref = torch.mm(q,k.transpose(-2,-1))
p_ref = torch.softmax(s_ref,dim=-1)
o_ref = torch.mm(p_ref,v)
max_ref,_ = torch.max(s_ref,dim=-1)
sum_ref = torch.exp(s_ref[:,0]) / p_ref[:,0]
sum_tensor = torch.exp(s_ref) / p_ref

#TODO: add scale d^-0.5
scale_d = q.shape[-1] ** (-0.5)
s_scaled = s_ref * scale_d
p_scaled = torch.softmax(s_scaled,dim=-1)
o_scaled = torch.mm(p_scaled,v)
sum_scaled = torch.exp(s_scaled[:,0]) / p_ref[:,0]
tmp = torch.exp(s_ref - max_ref.reshape(-1,1))
sum_scaled_test = torch.sum(torch.exp(s_ref - max_ref.reshape(-1,1)),dim=-1)

#TODO: declare fa data
o_fa = torch.zeros_like(q)
s_fa = torch.zeros(seq,seq,dtype=torch.float32)
max_fa = torch.zeros_like(max_ref)
sum_fa = torch.zeros_like(max_ref)

for it_Q in range(chunkQ):
    
    pre_sum = torch.zeros(tileQ,dtype=torch.float32)
    pre_max = torch.full_like(pre_sum,float("-inf"))
    
    for it_K in range(chunkK):
        score = torch.mm(q[it_Q*tileQ:(it_Q+1)*tileQ,:] ,k[it_K*tileK:(it_K+1)*tileK,:].transpose(-2,-1))
        s_fa[it_Q*tileQ:(it_Q+1)*tileQ,it_K*tileK:(it_K+1)*tileK] = score
        # 下面的写法会报错TypeError: max() received an invalid combination of arguments - got (torch.return_types.max, Tensor)
        # 原因是max函数返回最大值张量+索引两个张量，而实际只需要最大值这个张量
        # local_max= torch.max(score,dim=1)
         
        local_max,_ = torch.max(score,dim=-1)
        # 两个列向量求max值不会改变维度
        cur_max = torch.max(local_max, pre_max)
        
        # max函数对张量的某个维度求最大值会将低纬度压缩，因此再次与原张量运算需要reshape操作
        p = torch.exp((score - cur_max.reshape(-1,1))*scale_d)
        # p = torch.exp((score - cur_max.reshape(-1,1)))
        
        if it_K == 0:
            pre_sum = torch.sum(p,dim=-1)
            o_fa[it_Q*tileQ:(it_Q+1)*tileQ,:] = torch.mm(p,v[it_K*tileK:(it_K+1)*tileK,:])
        else:
            scale = torch.exp((pre_max - cur_max)*scale_d) 
            # scale = torch.exp((pre_max - cur_max)) 
            pre_sum = torch.sum(p,dim=-1) + pre_sum * scale
            o_fa[it_Q*tileQ:(it_Q+1)*tileQ,:] = torch.mm(p,v[it_K*tileK:(it_K+1)*tileK,:]) + o_fa[it_Q*tileQ:(it_Q+1)*tileQ,:] * scale.reshape(-1,1)
            
        pre_max = cur_max
        
    # 报错RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 2
    # 列向量，最后一维被压缩
    max_fa[it_Q*tileQ:(it_Q+1)*tileQ] = pre_max
    sum_fa[it_Q*tileQ:(it_Q+1)*tileQ] = pre_sum
    sum = pre_sum.reshape(-1,1)
    o_fa[it_Q*tileQ:(it_Q+1)*tileQ,:] = o_fa[it_Q*tileQ:(it_Q+1)*tileQ,:] / sum

#TODO: check
print(f"score max diff: {(s_ref - s_fa).abs().max().item()}")#pass
print(f"max max diff: {(max_ref - max_fa).abs().max().item()}")#pass
print(f"sum max diff: {(sum_scaled - sum_fa).abs().max().item()}")
a = sum_scaled / torch.exp(max_ref*scale_d)
print(f"out max diff: {(o_scaled - o_fa).abs().max().item()}")#pass

