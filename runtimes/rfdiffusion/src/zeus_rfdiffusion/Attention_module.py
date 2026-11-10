import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from opt_einsum import contract as einsum
from zeus_rfdiffusion.util_module import init_lecun_normal, symmetric_block_tile, profile_region


def _save_attn_tensor(attn, block, t, is_row, design_num=0):
    """Save post-softmax attention matrix for analysis.

    Controlled by env vars:
      RFDIFF_SAVE_ATTN=1 enables saving
      RFDIFF_ATTN_DIR=path sets output directory (default artifacts/attn_analysis)
      RFDIFF_ATTN_BLOCKS=comma,separated,block,names (default: extra_0,main_0,main_15,main_31,ref_0)
      RFDIFF_ATTN_TIMESTEPS=comma,separated,timesteps (default: 50,25,10,1)
      RFDIFF_ATTN_DESIGN_NUM=int (default 0)
    """
    if os.getenv("RFDIFF_SAVE_ATTN", "0") != "1":
        return
    if block is None or t is None:
        return
    blocks_env = os.getenv("RFDIFF_ATTN_BLOCKS", "extra_0,main_0,main_15,main_31,ref_0")
    save_all_blocks = blocks_env.strip().lower() == "all"
    save_blocks = set(blocks_env.split(",")) if not save_all_blocks else None
    save_timesteps = set(int(x) for x in os.getenv("RFDIFF_ATTN_TIMESTEPS", "50,25,10,1").split(","))
    t_val = t.item() if hasattr(t, 'item') else int(t)
    if (not save_all_blocks and block not in save_blocks) or t_val not in save_timesteps:
        return
    out_dir = os.getenv("RFDIFF_ATTN_DIR", "artifacts/attn_analysis")
    d_num = int(os.getenv("RFDIFF_ATTN_DESIGN_NUM", "0"))
    d_dir = os.path.join(out_dir, f"design_{d_num}")
    os.makedirs(d_dir, exist_ok=True)
    axis = "row" if is_row else "col"
    fpath = os.path.join(d_dir, f"attn_t{t_val}_block_{block}_{axis}.pt")
    torch.save(attn.detach().cpu().half(), fpath)

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model*r_ff, d_model)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before ReLu: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        # initialize linear layer right before residual connection: zero initialize
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, src):
        src = self.norm(src)
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src

class Attention(nn.Module):
    # calculate multi-head attention
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out):
        super(Attention, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        #
        self.to_q = nn.Linear(d_query, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head*d_hidden, bias=False)
        #
        self.to_out = nn.Linear(n_head*d_hidden, d_out)
        self.scaling = 1/math.sqrt(d_hidden)
        #
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]
        #
        query = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key = self.to_k(key).reshape(B, K, self.h, self.dim)
        value = self.to_v(value).reshape(B, K, self.h, self.dim)
        #
        query = query * self.scaling
        attn = einsum('bqhd,bkhd->bhqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bhqk,bkhd->bqhd', attn, value)
        out = out.reshape(B, Q, self.h*self.dim)
        #
        out = self.to_out(out)

        return out

class AttentionWithBias(nn.Module):
    def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32):
        super(AttentionWithBias, self).__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.norm_bias = nn.LayerNorm(d_bias)
        #
        self.to_q = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_in, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_in)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, bias):
        B, L = x.shape[:2]
        #
        x = self.norm_in(x)
        bias = self.norm_bias(bias)
        #
        query = self.to_q(x).reshape(B, L, self.h, self.dim)
        key = self.to_k(x).reshape(B, L, self.h, self.dim)
        value = self.to_v(x).reshape(B, L, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(x))
        #
        key = key * self.scaling
        attn = einsum('bqhd,bkhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bkhd->bqhd', attn, value).reshape(B, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

# MSA Attention (row/column) from AlphaFold architecture
class SequenceWeight(nn.Module):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super(SequenceWeight, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(self.dim)

        self.to_query = nn.Linear(d_msa, n_head*d_hidden)
        self.to_key = nn.Linear(d_msa, n_head*d_hidden)
        self.dropout = nn.Dropout(p_drop)

        self.reset_parameter()
    
    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
       
        tar_seq = msa[:,0]
        
        q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
        k = self.to_key(msa).view(B, N, L, self.h, self.dim)
        
        q = q * self.scale
        attn = einsum('bqihd,bkihd->bkihq', q, k)
        attn = F.softmax(attn, dim=1)
        return self.dropout(attn)

class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair): # TODO: make this as tied-attention
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        #
        seq_weight = self.seq_weight(msa) # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        key = key * self.scaling
        attn = einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

class EfficientMSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32, symm_n=1):
        super(EfficientMSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.symm_n = symm_n
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair, symm_n=1): # TODO: make this as tied-attention
        B, N, L = msa.shape[:3]
        if L % symm_n != 0:
            raise ValueError(f"MSA length {L} must be divisible by symm_n={symm_n}")
        L_asu = L // symm_n
        msa = msa[:, :, :L_asu, :]
        pair = pair[:, :L_asu, :, :]
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        #
        seq_weight = self.seq_weight(msa) # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim).repeat(1, 1, symm_n, 1, 1)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim).repeat(1, 1, symm_n, 1, 1)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(msa))
        #
        
        query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        key = key * self.scaling
        attn = einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        out = out.repeat(1, 1, symm_n, 1)
        return out

class RestrictedMSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32, symm_n=1):
        super(RestrictedMSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.symm_n = symm_n
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair, symm_n=1): # TODO: make this as tied-attention
        B, N, L = msa.shape[:3]
        _, LR, LC = pair.shape[:3]
        if L % symm_n != 0:
            raise ValueError(f"MSA length {L} must be divisible by symm_n={symm_n}")
        msa = msa[:, :, :LR, :]
        #
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        #
        seq_weight = self.seq_weight(msa) # (B, N, LR, h, 1)
        query = self.to_q(msa).reshape(B, N, LR, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, LR, self.h, self.dim).repeat(1, 1, symm_n, 1, 1)
        value = self.to_v(msa).reshape(B, N, LR, self.h, self.dim).repeat(1, 1, symm_n, 1, 1)
        bias = self.to_b(pair) # (B, LR, LC, h)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        key = key * self.scaling
        attn = einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, LR, -1)
        out = gate * out
        #
        out = self.to_out(out)
        out = out.repeat(1, 1, symm_n, 1)
        return out

class MSAColAttention(nn.Module):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(MSAColAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * self.scaling
        attn = einsum('bqihd,bkihd->bihqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihqk,bkihd->bqihd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

class EfficientMSAColAttention(nn.Module):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(EfficientMSAColAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, symm_n=1):
        B, N, L = msa.shape[:3]
        # Use full MSA — don't slice to ASU, let per-subunit features evolve
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * self.scaling
        attn = einsum('bqihd,bkihd->bihqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihqk,bkihd->bqihd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out


class MSAColGlobalAttention(nn.Module):
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(MSAColGlobalAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = query.mean(dim=1) # (B, L, h, dim)
        key = self.to_k(msa) # (B, N, L, dim)
        value = self.to_v(msa) # (B, N, L, dim)
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, h*dim)
        #
        query = query * self.scaling
        attn = einsum('bihd,bkid->bihk', query, key) # (B, L, h, N)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1) # (B, 1, L, h*dim)
        out = gate * out # (B, N, L, h*dim)
        #
        out = self.to_out(out)
        return out
    
class EfficientMSAColGlobalAttention(nn.Module):
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(EfficientMSAColGlobalAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, symm_n=1):
        B, N, L = msa.shape[:3]
        # Use full MSA — don't slice to ASU, let per-subunit features evolve
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = query.mean(dim=1) # (B, L, h, dim)
        key = self.to_k(msa) # (B, N, L, dim)
        value = self.to_v(msa) # (B, N, L, dim)
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, h*dim)
        #
        query = query * self.scaling
        attn = einsum('bihd,bkid->bihk', query, key) # (B, L, h, N)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1) # (B, 1, L, h*dim)
        out = gate * out # (B, N, L, h*dim)
        #
        out = self.to_out(out)
        return out

# Instead of triangle attention, use Tied axail attention with bias from coordinates..?
class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, block=None, t=None):
        B, L = pair.shape[:2]
        label = "BiasedAxialAttention.row" if self.is_row else "BiasedAxialAttention.col"
        with profile_region(label):
            if self.is_row:
                pair = pair.permute(0,2,1,3)
                bias = bias.permute(0,2,1,3)

            pair = self.norm_pair(pair)
            bias = self.norm_bias(bias)
            query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
            key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
            value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
            bias = self.to_b(bias)
            gate = torch.sigmoid(self.to_g(pair))

            query = query * self.scaling
            key = key / math.sqrt(L)
            attn = einsum('bnihk,bnjhk->bijh', query, key)
            attn = attn + bias
            attn = F.softmax(attn, dim=-2)
            _save_attn_tensor(attn, block, t, self.is_row)
            out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, L, L, -1)
            out = gate * out
            out = self.to_out(out)
            if self.is_row:
                out = out.permute(0,2,1,3)
        return out

class EfficientBiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(EfficientBiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, block=None, t=None, symm_n=1, perm_table=None, full=True):
        label = "EfficientBiasedAxialAttention.row" if self.is_row else "EfficientBiasedAxialAttention.col"
        with profile_region(label):
            if not full:
                B_in, LR_in, LC_in = pair.shape[:3]
                L_asu_in = LR_in
                pair = symmetric_block_tile(pair, perm_table if perm_table is not None else symm_n)
                bias = symmetric_block_tile(bias, perm_table if perm_table is not None else symm_n)

            if self.is_row:
                pair = pair.permute(0,2,1,3)
                bias = bias.permute(0,2,1,3)

            pair = self.norm_pair(pair)
            B, L = pair.shape[:2]
            L_asu = L // symm_n

            reduced_pair = pair[:, :, :L_asu, :]
            bias = self.norm_bias(bias)
            B, LR, LC = reduced_pair.shape[:3]
            B, PAIR_LR, PAIR_LC = pair.shape[:3]

            query = self.to_q(reduced_pair).reshape(B, LR, LC, self.h, self.dim)
            key = self.to_k(pair).reshape(B, PAIR_LR, PAIR_LC, self.h, self.dim)
            value = self.to_v(pair).reshape(B, PAIR_LR, PAIR_LC, self.h, self.dim)
            bias = self.to_b(bias)
            gate = torch.sigmoid(self.to_g(pair))

            query = query * self.scaling
            key = key / math.sqrt(L)
            attn_base = einsum('bnihk,bnjhk->bijh', query, key)
            attn = symmetric_block_tile(attn_base, perm_table if perm_table is not None else symm_n)
            attn = attn + bias
            attn = F.softmax(attn, dim=-2)
            out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, PAIR_LR, PAIR_LC, -1)
            out = gate * out
            out = self.to_out(out)

            if self.is_row:
                out = out.permute(0,2,1,3)
            if not full:
                out = out[:, :L_asu, :, :]
        return out

class RestrictedLiteBiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(RestrictedLiteBiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, block=None, t=None, symm_n=2, perm_table=None, full=False):
        # pair: (B, L_asu, L_full, d_pair)
        if not full:
            L_asu_in = pair.shape[1]
            _pt = perm_table if perm_table is not None else symm_n
            pair = symmetric_block_tile(pair, _pt)
            bias = symmetric_block_tile(bias, _pt)
        
        
        #Can we get rid of half of the pair feature dimension? (rows)
        pair = self.norm_pair(pair)
        B, L = pair.shape[:2]
        if L % symm_n != 0:
            raise ValueError(f"Pair length {L} must be divisible by symm_n={symm_n}")
        L_asu = L // symm_n
        
        #pair = pair[:, :L_asu, :, :]
        reduced_pair = pair[:, :, :L_asu, :]
        bias = self.norm_bias(bias)
        B, LR, LC = reduced_pair.shape[:3]
        B, PAIR_LR, PAIR_LC = pair.shape[:3]

        query = self.to_q(reduced_pair).reshape(B, LR, LC, self.h, self.dim)
        #Save query matrices for visualization
        # if block is not None and t is not None:
        #     torch.save(query, f'query/query_block{block}_t{t}.pt')
        key = self.to_k(pair).reshape(B, PAIR_LR, PAIR_LC, self.h, self.dim)
        value = self.to_v(pair).reshape(B, PAIR_LR, PAIR_LC, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
        query = query * self.scaling
        key = key / math.sqrt(L) # normalize for tied attention
        attn_base = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
        attn = symmetric_block_tile(attn_base, perm_table if perm_table is not None else symm_n)
        attn = attn + bias # apply bias
        attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, PAIR_LR, PAIR_LC, -1)
        out = gate * out
        out = self.to_out(out)
        
        #out = torch.cat([torch.roll(out, shifts=i*L_asu, dims=2) for i in range(symm_n)], dim=1)
        #save out matrix for visualization
        if not full:
            out = out[:, :L_asu, :, :]
        return out

class RestrictedBiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(RestrictedBiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, block=None, t=None, symm_n=None, perm_table=None):
        label = "RestrictedBiasedAxialAttention.row" if self.is_row else "RestrictedBiasedAxialAttention.col"
        with profile_region(label):
            return self._forward_impl(pair, bias, block, t, symm_n, perm_table)

    def _forward_impl(self, pair, bias, block=None, t=None, symm_n=None, perm_table=None):
        B, PAIR_LR, PAIR_LC = pair.shape[:3]
        LR = PAIR_LR
        LC = PAIR_LC
        symm_n = LC // LR

        # Build a cyclic perm_table if not provided (backward-compat for
        # cyclic callers). perm_table[g, h] = index of g⁻¹h in the group.
        # For cyclic C_n, this is (h - g) mod n — produces identical indexing
        # as the old modular-arithmetic code.
        if perm_table is None:
            perm_table = torch.stack([
                torch.roll(torch.arange(symm_n), shifts=g) for g in range(symm_n)
            ])
        # perm_table is a small constant integer table but lives on the GPU.
        # Indexing it element-wise in the loops below costs a device sync per
        # lookup (~3*symm_n^2 per call), which dominates runtime. Read it out
        # as Python ints once.
        perm = perm_table.tolist()      # S x S nested list
        inv = [row[0] for row in perm]  # inv[g] = index of g⁻¹

        # === RESTRICTED: max intermediate size is (LR, LC), never (LC, LC) ===
        # For is_row: block-row 0 of pair^T. In a symmetric assembly:
        #   pair^T[r, c] = pair[c, r]. For block-row 0 of pair^T at block-col b:
        #   block [0, b] of pair^T = block [b, 0] of pair = pair[0, g⁻¹·0] = pair[0, inv(b)]
        # So start (in residue indices) = inv_table[b] * LR.
        if self.is_row:
            pair_r = torch.empty_like(pair)
            bias_r = torch.empty_like(bias)
            for b in range(symm_n):
                start = inv[b] * LR
                pair_r[:, :, b*LR:(b+1)*LR, :] = pair[:, :, start:start+LR, :].permute(0, 2, 1, 3)
                bias_r[:, :, b*LR:(b+1)*LR, :] = bias[:, :, start:start+LR, :].permute(0, 2, 1, 3)
        else:
            pair_r = pair   # (B, LR, LC, D)
            bias_r = bias

        pair_r = self.norm_pair(pair_r)
        bias_r = self.norm_bias(bias_r)
        Q_r = self.to_q(pair_r).reshape(B, LR, LC, self.h, self.dim)
        K_r = self.to_k(pair_r).reshape(B, LR, LC, self.h, self.dim)
        V_r = self.to_v(pair_r).reshape(B, LR, LC, self.h, self.dim)
        bias_proj_r = self.to_b(bias_r)            # (B, LR, LC, h)
        gate_r = torch.sigmoid(self.to_g(pair_r))  # (B, LR, LC, h*dim)

        Q_r = Q_r * self.scaling
        K_r = K_r / math.sqrt(LC)

        # Split Q, K into symm_n column-blocks of size LR: each (B, LR, LR, h, dim)
        Q_blocks = list(Q_r.chunk(symm_n, dim=2))
        K_blocks = list(K_r.chunk(symm_n, dim=2))

        # Attention logits for queries in block-row 0. Full tied attn at block
        # (qi, kj) sums over block-rows k: Q_full[k, qi]^T K_full[k, kj].
        # In restricted (symmetric) form: Q_full[k, qi] = Q_r[0, perm[k, qi]].
        # For qi=0: sum_k Q_blocks[perm[k, 0]]^T @ K_blocks[perm[k, kj]].
        attn_logit_blocks = []  # symm_n blocks of (B, LR, LR, h)
        for kj in range(symm_n):
            block_sum = torch.zeros(B, LR, LR, self.h, device=pair.device)
            for k in range(symm_n):
                qi_idx = perm[k][0]   # = inv(k)
                kj_idx = perm[k][kj]  # = k⁻¹ · kj
                block_sum = block_sum + einsum(
                    'bnihk,bnjhk->bijh', Q_blocks[qi_idx], K_blocks[kj_idx]
                )
            attn_logit_blocks.append(block_sum)
        attn_logits_r = torch.cat(attn_logit_blocks, dim=2)  # (B, LR, LC, h)
        attn_logits_r = attn_logits_r + bias_proj_r
        attn_r = F.softmax(attn_logits_r, dim=-2)            # (B, LR, LC, h)

        V_blocks = list(V_r.chunk(symm_n, dim=2))    # each (B, LR, LR, h, dim)
        attn_r_blocks = list(attn_r.chunk(symm_n, dim=2))  # each (B, LR, LR, h)

        if not self.is_row:
            # Col attn output at block cb uses V_blocks[perm[cb, kj]].
            out_col_blocks = []
            for cb in range(symm_n):
                out_cb = torch.zeros(B, LR, LR, self.h, self.dim, device=pair.device)
                for kj in range(symm_n):
                    v_idx = perm[cb][kj]
                    out_cb = out_cb + einsum(
                        'bijh,bkjhd->bikhd', attn_r_blocks[kj], V_blocks[v_idx]
                    )
                out_col_blocks.append(out_cb)
            restricted_out = torch.cat(out_col_blocks, dim=2).reshape(B, LR, LC, -1)
            restricted_out = gate_r * restricted_out
            restricted_out = self.to_out(restricted_out)
        else:
            # Row attn output at block bi uses V_blocks[(kj+bi) % S] in cyclic;
            # in general: V_blocks[perm[inv(bi), kj]], and gate block uses inv(bi).
            gate_r_blocks = list(gate_r.chunk(symm_n, dim=2))  # each (B, LR, LR, h*dim)
            out_col_blocks = []
            for bi in range(symm_n):
                inv_bi = inv[bi]
                out_bi = torch.zeros(B, LR, LR, self.h, self.dim, device=pair.device)
                for kj in range(symm_n):
                    v_idx = perm[inv_bi][kj]
                    out_bi = out_bi + einsum(
                        'bijh,bkjhd->bkihd', attn_r_blocks[kj], V_blocks[v_idx]
                    )
                out_bi = out_bi.reshape(B, LR, LR, -1)  # (B, LR_k, LR_i, h*dim)
                gate_bi = gate_r_blocks[inv_bi].permute(0, 2, 1, 3)  # (B, LR_k, LR_i, h*dim)
                out_bi = gate_bi * out_bi
                out_col_blocks.append(out_bi)
            restricted_out = torch.cat(out_col_blocks, dim=2).reshape(B, LR, LC, -1)
            restricted_out = self.to_out(restricted_out)  # single call

        return restricted_out