import os
import torch.utils.checkpoint as checkpoint
from zeus_rfdiffusion.util_module import *
from zeus_rfdiffusion.Attention_module import *
from zeus_rfdiffusion.SE3_network import SE3TransformerWrapper



# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

# Update MSA with biased self-attention. bias from Pair & Str
class EfficientMSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(EfficientMSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = EfficientMSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = EfficientMSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = EfficientMSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state, symm_n=1):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair) # (B, L, L, d_pair)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair, symm_n=symm_n))
        msa = msa + self.col_attn(msa, symm_n=symm_n)
        msa = msa + self.ff(msa)

        return msa

class RestrictedMSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(RestrictedMSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = RestrictedMSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = EfficientMSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) #Efficient/REstricted is the same
        else:
            self.col_attn = EfficientMSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) #Efficient/Restricted is the same since we do not use pair
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state, symm_n=1):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair) # (B, L, L, d_pair)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair, symm_n=symm_n))
        msa = msa + self.col_attn(msa, symm_n=symm_n)
        msa = msa + self.ff(msa)

        return msa


class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state, symm_n=None):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair) # (B, L, L, d_pair)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa

class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(PairStr2Pair, self).__init__()

        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_rbf.bias)

        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat, block=None, t=None, symm_n=None, perm_table=None):
        B, L = pair.shape[:2]

        rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))


        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat, block=block, t=t))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat, block=block, t=t))
        pair = pair + self.ff(pair)
        return pair

class EfficientPairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(EfficientPairStr2Pair, self).__init__()
        
        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = EfficientBiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = EfficientBiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_rbf.bias)
        
        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat, block=None, t=None, symm_n=1, perm_table=None):
        with profile_region("EfficientPairStr2Pair"):
            rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))
            pair = pair + self.drop_row(self.row_attn(pair, rbf_feat, block=block, t=t, symm_n=symm_n, perm_table=perm_table, full=False))
            pair = pair + self.drop_col(self.col_attn(pair, rbf_feat, block=block, t=t, symm_n=symm_n, perm_table=perm_table, full=False))
            pair = pair + self.ff(pair)
        return pair

class RestrictedPairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(RestrictedPairStr2Pair, self).__init__()

        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = RestrictedBiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = RestrictedBiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.emb_rbf.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_rbf.bias)

        self.proj_rbf = init_lecun_normal(self.proj_rbf)
        nn.init.zeros_(self.proj_rbf.bias)

    def forward(self, pair, rbf_feat, block=None, t=None, symm_n=None, perm_table=None):
        rbf_feat = self.proj_rbf(F.relu_(self.emb_rbf(rbf_feat)))

        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat, block=block, t=t, perm_table=perm_table))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat, block=block, t=t, perm_table=perm_table))
        pair = pair + self.ff(pair)
        return pair
class EfficientMSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(EfficientMSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair, symm_n=1):
        B, LR, LC = pair.shape[:3]
        if LR % symm_n != 0:
            raise ValueError(f"Pair length {LR} must be divisible by symm_n={symm_n}")
        L_asu = LR // symm_n
        msa = msa[:, :, :L_asu, :]
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
        out = out.repeat(1, symm_n, symm_n, 1)
        pair = pair + out
        return pair

class RestrictedMSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(RestrictedMSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair, symm_n=1):
        B, LR, LC = pair.shape[:3]
        symm_n = LC//LR
        if LC % LR != 0:
            raise ValueError(f"Pair length {LC} must be divisible by symm_n={symm_n}")
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
        out = out[:, :LR, :LC, :]
        pair = pair + out
        return pair

class EfficientRestrictedMSA2Pair(nn.Module):
    """Like RestrictedMSA2Pair but computes the outer product only for
    the restricted rows/cols: O(LR * LC) instead of O(L²).
    Same weights, drop-in replacement."""
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(EfficientRestrictedMSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair, symm_n=1):
        B, LR, LC = pair.shape[:3]
        symm_n = LC // LR
        if LC % LR != 0:
            raise ValueError(f"Pair length {LC} must be divisible by symm_n={symm_n}")
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)[:, :, :LR, :]   # (B, N, LR, d_hidden)
        right = self.proj_right(msa)[:, :, :LC, :]  # (B, N, LC, d_hidden)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, LR, LC, -1)
        out = self.proj_out(out)
        pair = pair + out
        return pair


class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair, symm_n=None):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)#.repeat(1,1,2,1)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1) #Should be [1,100,100,256]
        out = self.proj_out(out)
       
        pair = pair + out
        
        return pair

class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.linear_out = nn.Linear(d_hidden, 20)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)
        
        # right before relu activation: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # right before residual connection: zero initialize
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq, state):
        '''
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: hidden embeddings corresponding to query sequence (B, L, d_msa)
            - state: state feature (output l0 feature) from previous SE3 layer (B, L, d_state)
        Outputs:
            - si: predicted torsion angles (phi, psi, omega, chi1~4 with cos/sin, Cb bend, Cb twist, CG) (B, L, 10, 2)
        '''
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        si = self.linear_out(F.relu_(si))
        return si.view(B, L, 10, 2)


class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
    
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                                   p_drop=p_drop)
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, symm_n=1, cyclic_reses=None, top_k=64, eps=1e-5, perm_table=None, sym_rots=None):
        B, N, L = msa.shape[:3]

        if motif_mask is None:
            motif_mask = torch.zeros(L).bool()

        # process msa & pair features
        node = self.norm_msa(msa[:,0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)
       
        node = torch.cat((node, state), dim=-1)
        node = self.norm_node(self.embed_x(node))
        pair = self.norm_edge1(self.embed_e1(pair))
        
        neighbor = get_seqsep(idx, cyclic_reses)
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1]))
        pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))


        #Now we need to modify this graph procedure. Restrict msa nodes to 1/2, but connect 
        # define graph
        if top_k != 0:
            G, edge_feats = make_topk_graph(xyz[:,:,1,:], pair, idx, top_k=top_k)
        else:
            G, edge_feats = make_full_graph(xyz[:,:,1,:], pair, idx, top_k=top_k)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)

        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, node.reshape(B*L, -1, 1), l1_feats, edge_feats)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3)
        offset[:,motif_mask,...] = 0            # NOTE: motif mask is all zeros if not freeezing the motif 

        delTi = offset[:,:,0,:] / 10.0 # translation
        R = offset[:,:,1,:] / 100.0 # rotation
        
        Qnorm = torch.sqrt( 1 + torch.sum(R*R, dim=-1) )
        qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        delRi = torch.zeros((B,L,3,3), device=xyz.device)
        delRi[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        delRi[:,:,0,1] = 2*qB*qC - 2*qA*qD
        delRi[:,:,0,2] = 2*qB*qD + 2*qA*qC
        delRi[:,:,1,0] = 2*qB*qC + 2*qA*qD
        delRi[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        delRi[:,:,1,2] = 2*qC*qD - 2*qA*qB
        delRi[:,:,2,0] = 2*qB*qD - 2*qA*qC
        delRi[:,:,2,1] = 2*qC*qD + 2*qA*qB
        delRi[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD

        Ri = einsum('bnij,bnjk->bnik', delRi, R_in)
        Ti = delTi + T_in #einsum('bnij,bnj->bni', delRi, T_in) + delTi
            
        alpha = self.sc_predictor(msa[:,0], state)
        return Ri, Ti, state, alpha

class RestrictedStr2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(RestrictedStr2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
    
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                                   p_drop=p_drop)
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, symm_n=1, cyclic_reses=None, top_k=64, eps=1e-5, perm_table=None, sym_rots=None):
        B, N, L = msa.shape[:3]
        B, LR, LC = pair.shape[:3]
        symm_n = LC // LR

        if motif_mask is None:
            motif_mask = torch.zeros(L).bool()

        # Node features (O(L), no pair involved)
        node = self.norm_msa(msa[:,0])
        state = self.norm_state(state)
        node = torch.cat((node, state), dim=-1)
        node = self.norm_node(self.embed_x(node))

        # Pair embeddings at RESTRICTED size (LR, LC) — O(L_asu * L), not O(L²)
        pair = self.norm_pair(pair)
        pair = self.norm_edge1(self.embed_e1(pair))                           # (B, LR, LC, 32)
        rbf_restricted = rbf(torch.cdist(xyz[:, :LR, 1], xyz[:, :, 1]))      # (B, LR, LC, 36)
        neighbor_restricted = get_seqsep(idx, cyclic_reses)[:, :LR, :, :]    # (B, LR, LC, 1)
        pair = torch.cat((pair, rbf_restricted, neighbor_restricted), dim=-1) # (B, LR, LC, 69)
        pair = self.norm_edge2(self.embed_e2(pair))                           # (B, LR, LC, 32)

        import numpy as _np
        from se3_transformer.model.basis import get_basis, update_basis_with_fused
        from se3_transformer.model.transformer import get_populated_edge_features

        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)
        node_flat = node.reshape(B*L, -1, 1)

        attn_block = self.se3.se3.graph_modules[0]
        norm_layer = self.se3.se3.graph_modules[1]
        final_conv = self.se3.se3.graph_modules[2]

        def _build_basis(G):
            ef_pop = get_populated_edge_features(G.edata['rel_pos'], {'0': G.edata['edge_feats']})
            bas = get_basis(G.edata['rel_pos'], max_degree=self.se3.se3.max_degree,
                            compute_gradients=False, use_pad_trick=False, amp=torch.is_autocast_enabled())
            bas = update_basis_with_fused(bas, self.se3.se3.max_degree, use_pad_trick=False, fully_fused=False)
            return ef_pop, bas

        def _tile_asu(feat_dict, L_asu, group_rots):
            """Tile ASU features to full: deg0=repeat, deg1=rotate by group_rots[k].

            Args:
                feat_dict: {deg_str: Tensor} — ASU-only features (N=L_asu).
                L_asu: ASU size.
                group_rots: (S, 3, 3) rotation matrices. Identity at index 0.
            """
            S = group_rots.shape[0]
            out = {}
            for deg_str, feat in feat_dict.items():
                deg = int(deg_str)
                if deg == 0:
                    out[deg_str] = feat.repeat(S, 1, 1)
                elif deg == 1:
                    blocks = [feat]  # k=0: identity
                    for k in range(1, S):
                        R_k = group_rots[k]  # (3, 3)
                        blocks.append(torch.einsum('nci,ij->ncj', feat, R_k.T))
                    out[deg_str] = torch.cat(blocks, dim=0)
            return out

        if symm_n > 1:
            L_asu = LR
            # Prefer caller-provided sym_rots (e.g., tetrahedral, octahedral).
            # Fall back to cyclic-Z rotations for backwards compat.
            if sym_rots is not None:
                group_rots = sym_rots.to(xyz.device).float()
            else:
                angle = 2 * _np.pi / symm_n
                c, s = _np.cos(angle), _np.sin(angle)
                R1 = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],
                                  dtype=torch.float32, device=xyz.device)
                group_rots = torch.stack([
                    torch.matrix_power(R1, k) for k in range(symm_n)
                ])

            # Build ASU-dest graph directly: O(L*L_asu) instead of O(L²)
            if top_k != 0:
                # Use the `_matched` variant whose edge set matches make_topk_graph
                # (the one Str2Str uses). The older make_asu_dest_topk_graph
                # introduced extra Direction-1 edges + restricted-pool top-k that
                # put the trained SE3 out-of-distribution for polyhedral groups.
                G_asu, edge_feats_asu = make_asu_dest_topk_graph_matched(
                    xyz[:,:,1,:], pair, idx, L_asu, top_k=top_k, perm_table=perm_table)
            else:
                G_asu, edge_feats_asu = make_asu_dest_full_graph(
                    xyz[:,:,1,:], pair, idx, L_asu, perm_table=perm_table)
            G_asu.edata['edge_feats'] = edge_feats_asu
            ef_pop_asu, basis_asu = _build_basis(G_asu)

            # Step 1: Attention on ASU-dest graph (all L nodes have features,
            #         but only ASU nodes receive messages)
            node_feats = {'0': node_flat, '1': l1_feats}
            attn_out_raw = attn_block(node_feats, ef_pop_asu, graph=G_asu, basis=basis_asu)
            attn_out_asu = {k: v[:L_asu] for k, v in attn_out_raw.items()}

            # Step 2: Tile ASU attention output to full
            attn_tiled = _tile_asu(attn_out_asu, L_asu, group_rots)

            # Step 3: NormSE3 (pointwise, no graph needed)
            normed = norm_layer(attn_tiled)

            # Step 4: ConvSE3 on ASU-dest graph with tiled normed features
            shift_raw = final_conv(normed, ef_pop_asu, graph=G_asu, basis=basis_asu)
            shift_asu = {k: v[:L_asu] for k, v in shift_raw.items()}

            # Step 5: Tile conv output to full
            shift = _tile_asu(shift_asu, L_asu, group_rots)

            # Validation: compare direct ASU-dest vs filtered-from-full-graph
            if os.getenv("RFDIFF_VALIDATE_ASU", "0") == "1":
                if top_k != 0:
                    pair_tiled = symmetric_block_tile(pair, perm_table if perm_table is not None else symm_n)
                    G_full, ef_full = make_topk_graph(xyz[:,:,1,:], pair_tiled, idx, top_k=top_k)
                else:
                    G_full, ef_full = make_full_graph_from_restricted(xyz[:,:,1,:], pair, idx, LR, perm_table=perm_table)
                G_full.edata['edge_feats'] = ef_full
                ef_pop_full, basis_full = _build_basis(G_full)

                node_feats_full = {'0': node_flat, '1': l1_feats}
                attn_out_full = attn_block(node_feats_full, ef_pop_full, graph=G_full, basis=basis_full)
                _block_tag = os.getenv("RFDIFF_SE3_BLOCK", "?")
                _t_tag = os.getenv("RFDIFF_SE3_TIMESTEP", "?")
                for ds in attn_out_asu:
                    ref = attn_out_full[ds][:L_asu]
                    dev = (attn_out_asu[ds] - ref).abs().max().item()
                    rel = dev / (ref.abs().max().item() + 1e-10)
                    print(f"  [t={_t_tag} {_block_tag}] VALIDATE attn deg{ds}: abs={dev:.4e} rel={rel:.4e}")

                # Full conv with symmetrized full-attention features
                attn_sym = _tile_asu({k: v[:L_asu] for k, v in attn_out_full.items()}, L_asu, group_rots)
                normed_full = norm_layer(attn_sym)
                shift_full = final_conv(normed_full, ef_pop_full, graph=G_full, basis=basis_full)
                for ds in shift_asu:
                    ref = shift_full[ds][:L_asu]
                    dev = (shift_asu[ds] - ref).abs().max().item()
                    rel = dev / (ref.abs().max().item() + 1e-10)
                    print(f"  [t={_t_tag} {_block_tag}] VALIDATE conv deg{ds}: abs={dev:.4e} rel={rel:.4e}")

        else:
            # Non-symmetric: build full graph, run SE3 normally
            if top_k != 0:
                G_full, ef_full = make_topk_graph(xyz[:,:,1,:], pair, idx, top_k=top_k)
            else:
                G_full, ef_full = make_full_graph_from_restricted(xyz[:,:,1,:], pair, idx, LR)
            G_full.edata['edge_feats'] = ef_full
            ef_pop_full, basis_full = _build_basis(G_full)
            node_feats = {'0': node_flat, '1': l1_feats}
            attn_out = attn_block(node_feats, ef_pop_full, graph=G_full, basis=basis_full)
            normed = norm_layer(attn_out)
            shift = final_conv(normed, ef_pop_full, graph=G_full, basis=basis_full)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)

        offset = shift['1'].reshape(B, L, 2, 3)
        offset[:,motif_mask,...] = 0            # NOTE: motif mask is all zeros if not freeezing the motif

        delTi = offset[:,:,0,:] / 10.0 # translation
        R = offset[:,:,1,:] / 100.0 # rotation
        
        Qnorm = torch.sqrt( 1 + torch.sum(R*R, dim=-1) )
        qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        delRi = torch.zeros((B,L,3,3), device=xyz.device)
        delRi[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        delRi[:,:,0,1] = 2*qB*qC - 2*qA*qD
        delRi[:,:,0,2] = 2*qB*qD + 2*qA*qC
        delRi[:,:,1,0] = 2*qB*qC + 2*qA*qD
        delRi[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        delRi[:,:,1,2] = 2*qC*qD - 2*qA*qB
        delRi[:,:,2,0] = 2*qB*qD - 2*qA*qC
        delRi[:,:,2,1] = 2*qC*qD + 2*qA*qB
        delRi[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD

        Ri = einsum('bnij,bnjk->bnik', delRi, R_in)
        Ti = delTi + T_in #einsum('bnij,bnj->bni', delRi, T_in) + delTi
            
        alpha = self.sc_predictor(msa[:,0], state)
        return Ri, Ti, state, alpha


class EfficientRestrictedStr2Str(nn.Module):
    """Memory-efficient Str2Str for symmetric oligomers.

    Instead of tiling pair to (L, L) and building an L²-edge graph, this:
    1. Computes pair embeddings at restricted size (LR, LC) — no O(L²) pair
    2. Builds ASU-dest graph (L_asu * L edges instead of L²)
    3. Runs AttentionBlock on ASU graph → correct output at ASU nodes
    4. Tiles intermediate features to all L nodes using Wigner D-matrices
    5. Runs NormSE3 + final ConvSE3 on ASU graph

    Shares weights with RestrictedStr2Str — identical architecture, just
    different forward pass that avoids O(L²) memory.
    """
    def __init__(self, d_msa=256, d_pair=128, d_state=16,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(EfficientRestrictedStr2Str, self).__init__()

        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)

        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])

        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])

        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                                   p_drop=p_drop)

        # Cached Wigner D-matrices: computed once per symmetry order
        self._cached_symm_n = None
        self._cached_D = None  # dict: {degree: D1_matrix}

        self.reset_parameter()

    def reset_parameter(self):
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)

    def _get_wigner_d(self, symm_n, device):
        """Get or compute Wigner D-matrices for the given symmetry order."""
        if self._cached_symm_n == symm_n and self._cached_D is not None:
            return self._cached_D

        import numpy as np
        angle = 2 * np.pi / symm_n
        c, s = np.cos(angle), np.sin(angle)

        D = {}
        D[0] = torch.eye(1, device=device)
        D[1] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                             dtype=torch.float32, device=device)
        # Compute D2 on CPU (cusolver can fail on GPU during inference)
        D[2] = compute_wigner_d2(angle, device='cpu').to(device)

        self._cached_symm_n = symm_n
        self._cached_D = D
        return D

    def _tile_with_wigner(self, features_dict, L_asu, symm_n, D1_per_degree):
        """Tile ASU node features to all subunits using Wigner D-matrices."""
        tiled = {}
        for deg_str, feat in features_dict.items():
            deg = int(deg_str)
            asu_feat = feat[:L_asu]
            blocks = [asu_feat]
            D_k = D1_per_degree[deg].clone()
            for k in range(1, symm_n):
                if k == 1:
                    blocks.append(torch.einsum('nci,ij->ncj', asu_feat, D_k))
                else:
                    D_k = D_k @ D1_per_degree[deg]
                    blocks.append(torch.einsum('nci,ij->ncj', asu_feat, D_k))
            tiled[deg_str] = torch.cat(blocks, dim=0)
        return tiled

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, symm_n=1, cyclic_reses=None, top_k=64, eps=1e-5, perm_table=None, sym_rots=None):
        B, N, L = msa.shape[:3]
        B, LR, LC = pair.shape[:3]
        symm_n = LC // LR
        L_asu = LR

        if motif_mask is None:
            motif_mask = torch.zeros(L).bool()

        # --- Node features: same as RestrictedStr2Str (O(L), no pair involved) ---
        node = self.norm_msa(msa[:, 0])
        state = self.norm_state(state)
        node = torch.cat((node, state), dim=-1)
        node = self.norm_node(self.embed_x(node))

        # --- Pair embeddings at RESTRICTED size (LR, LC) — O(L_asu * L) not O(L²) ---
        pair = self.norm_pair(pair)                           # (B, LR, LC, d_pair)
        pair = self.norm_edge1(self.embed_e1(pair))           # (B, LR, LC, 32)

        neighbor_restricted = get_seqsep(idx, cyclic_reses)[:, :L_asu, :, :]  # (B, LR, L, 1)
        rbf_restricted = rbf(torch.cdist(xyz[:, :L_asu, 1], xyz[:, :, 1]))    # (B, LR, L, 36)
        pair = torch.cat((pair, rbf_restricted, neighbor_restricted), dim=-1)  # (B, LR, LC, 69)
        pair = self.norm_edge2(self.embed_e2(pair))           # (B, LR, LC, 32)

        # --- ASU-dest graph: only edges to ASU nodes ---
        G_asu, edge_feats = make_asu_full_graph(xyz[:, :, 1, :], pair, idx, L_asu)

        l1_feats = xyz - xyz[:, :, 1, :].unsqueeze(2)
        l1_feats = l1_feats.reshape(B * L, -1, 3)

        # --- SE3 with tile-between-layers ---
        from se3_transformer.model.basis import get_basis, update_basis_with_fused
        from se3_transformer.model.transformer import get_populated_edge_features

        node_feats = {'0': node.reshape(B * L, -1, 1), '1': l1_feats}
        edge_feats_pop = get_populated_edge_features(G_asu.edata['rel_pos'], {'0': edge_feats})
        basis = get_basis(G_asu.edata['rel_pos'], max_degree=self.se3.se3.max_degree,
                          compute_gradients=False, use_pad_trick=False, amp=torch.is_autocast_enabled())
        basis = update_basis_with_fused(basis, self.se3.se3.max_degree, use_pad_trick=False, fully_fused=False)

        attn_block = self.se3.se3.graph_modules[0]   # AttentionBlockSE3
        norm_layer = self.se3.se3.graph_modules[1]    # NormSE3
        final_conv = self.se3.se3.graph_modules[2]    # ConvSE3

        # Step 1: AttentionBlock on ASU graph
        attn_out = attn_block(node_feats, edge_feats_pop, graph=G_asu, basis=basis)

        # Step 2: Tile intermediate features with Wigner D-matrices
        D = self._get_wigner_d(symm_n, device=xyz.device)
        tiled = self._tile_with_wigner(attn_out, L_asu, symm_n, D)

        # Step 3: NormSE3 on tiled features
        normed = norm_layer(tiled)

        # Step 4: Final ConvSE3 on ASU graph (source features now correct)
        shift = final_conv(normed, edge_feats_pop, graph=G_asu, basis=basis)

        # --- Unpack output: only ASU nodes have values ---
        # Degree 0 (scalars): identical across subunits, simple repeat
        state_out = shift['0'][:L_asu].reshape(B, L_asu, -1).repeat(1, symm_n, 1)  # (B, L, C)
        # Degree 1 (3D vectors): rotate by Cn symmetry rotation for each subunit
        offset_asu = shift['1'][:L_asu].reshape(B, L_asu, 2, 3)  # (B, L_asu, 2, 3)
        offset_blocks = [offset_asu]
        for k in range(1, symm_n):
            D_k = torch.matrix_power(D[1], k)  # 3x3 rotation matrix
            offset_blocks.append(torch.einsum('bnvi,ij->bnvj', offset_asu, D_k))
        offset = torch.cat(offset_blocks, dim=1)  # (B, L, 2, 3)
        offset[:, motif_mask, ...] = 0

        delTi = offset[:, :, 0, :] / 10.0
        R = offset[:, :, 1, :] / 100.0

        Qnorm = torch.sqrt(1 + torch.sum(R * R, dim=-1))
        qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        delRi = torch.zeros((B, L, 3, 3), device=xyz.device)
        delRi[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        delRi[:,:,0,1] = 2*qB*qC - 2*qA*qD
        delRi[:,:,0,2] = 2*qB*qD + 2*qA*qC
        delRi[:,:,1,0] = 2*qB*qC + 2*qA*qD
        delRi[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        delRi[:,:,1,2] = 2*qC*qD - 2*qA*qB
        delRi[:,:,2,0] = 2*qB*qD - 2*qA*qC
        delRi[:,:,2,1] = 2*qC*qD + 2*qA*qB
        delRi[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD

        Ri = einsum('bnij,bnjk->bnik', delRi, R_in)
        Ti = delTi + T_in

        alpha = self.sc_predictor(msa[:, 0], state_out)
        return Ri, Ti, state_out, alpha


def _adapt_pair_for_str2str(pair, symmetry_mode, symm_n, perm_table):
    """Adapter between the pair output of pair2pair and the input of str2str.

    Three modes:
      - 'restricted'         : pair is already restricted (B, L_asu, L, D) and
                              str2str is RestrictedStr2Str — no change.
      - 'hybrid_restr_str'   : pair is full (B, L, L, D) but str2str is
                              RestrictedStr2Str — slice block-row 0.
      - 'hybrid_restr_pair'  : pair is restricted (B, L_asu, L, D) but str2str
                              is the null Str2Str — tile to full via perm_table.
      - None / null           : pair is full and str2str is null — no change.
    """
    if symm_n <= 1:
        return pair
    if symmetry_mode == 'hybrid_restr_str':
        L_asu = pair.shape[1] // symm_n
        return pair[:, :L_asu, :, :]
    if symmetry_mode == 'hybrid_restr_pair':
        # Tile restricted pair (LR=L_asu, LC=L) to full (L, L) using perm_table.
        return symmetric_block_tile(pair, perm_table if perm_table is not None else symm_n)
    return pair


class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 symmetry_mode=None):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.symmetry_mode = symmetry_mode
        if symmetry_mode == 'restricted':
            self.msa2msa = RestrictedMSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)
            self.msa2pair = EfficientRestrictedMSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                                        d_hidden=d_hidden//2, p_drop=p_drop)
            self.pair2pair = RestrictedPairStr2Pair(d_pair=d_pair, n_head=n_head_pair,
                                                    d_hidden=d_hidden, p_drop=p_drop)
            self.str2str = RestrictedStr2Str(d_msa=d_msa, d_pair=d_pair,
                                             d_state=SE3_param['l0_out_features'],
                                             SE3_param=SE3_param,
                                             p_drop=p_drop)
        elif symmetry_mode == 'hybrid_restr_str':
            # Isolation experiment: null for MSA/pair paths, restricted for str2str.
            # Pair stays full (L, L, D); str2str's input is sliced to block-row 0
            # in forward.
            self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                            n_head=n_head_msa,
                            d_state=SE3_param['l0_out_features'],
                            use_global_attn=use_global_attn,
                            d_hidden=d_hidden_msa, p_drop=p_drop)
            self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                     d_hidden=d_hidden//2, p_drop=p_drop)
            self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair,
                                          d_hidden=d_hidden, p_drop=p_drop)
            self.str2str = RestrictedStr2Str(d_msa=d_msa, d_pair=d_pair,
                                             d_state=SE3_param['l0_out_features'],
                                             SE3_param=SE3_param,
                                             p_drop=p_drop)
        elif symmetry_mode == 'hybrid_restr_pair':
            # Reverse isolation experiment: restricted for MSA/pair paths,
            # null for str2str. Pair stays restricted (L_asu, L, D);
            # before str2str, pair is tiled to full (L, L, D) via
            # symmetric_block_tile(perm_table).
            self.msa2msa = RestrictedMSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)
            self.msa2pair = EfficientRestrictedMSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                                        d_hidden=d_hidden//2, p_drop=p_drop)
            self.pair2pair = RestrictedPairStr2Pair(d_pair=d_pair, n_head=n_head_pair,
                                                    d_hidden=d_hidden, p_drop=p_drop)
            self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                   d_state=SE3_param['l0_out_features'],
                                   SE3_param=SE3_param,
                                   p_drop=p_drop)
        else:
            self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                            n_head=n_head_msa,
                            d_state=SE3_param['l0_out_features'],
                            use_global_attn=use_global_attn,
                            d_hidden=d_hidden_msa, p_drop=p_drop)
            self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                     d_hidden=d_hidden//2, p_drop=p_drop)
            self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair,
                                          d_hidden=d_hidden, p_drop=p_drop)
            self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                   d_state=SE3_param['l0_out_features'],
                                   SE3_param=SE3_param,
                                   p_drop=p_drop)

    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, block=None, t=None, symm_n=1, perm_table=None, sym_rots=None, use_checkpoint=False, cyclic_reses=None):
        B, LR, LC = pair.shape[:3]
        rbf_feat = rbf(torch.cdist(xyz[:,:LR,1,:], xyz[:,:LC,1,:]))

        if use_checkpoint:
            with profile_region("IterBlock.msa2msa"):
                msa = checkpoint.checkpoint(
                    create_custom_forward(self.msa2msa, symm_n=symm_n), msa, pair, rbf_feat, state
                )
            with profile_region("IterBlock.msa2pair"):
                pair = checkpoint.checkpoint(
                    create_custom_forward(self.msa2pair, symm_n=symm_n), msa, pair
                )
            with profile_region("IterBlock.pair2pair"):
                pair = checkpoint.checkpoint(
                    create_custom_forward(self.pair2pair, block=block, t=t, symm_n=symm_n, perm_table=perm_table), pair, rbf_feat
                )
            # Hybrid adapters between pair and str2str (see _adapt_pair_for_str2str).
            pair_for_str2str = _adapt_pair_for_str2str(pair, self.symmetry_mode, symm_n, perm_table)
            with profile_region("IterBlock.str2str"):
                R, T, state, alpha = checkpoint.checkpoint(
                    create_custom_forward(self.str2str, top_k=0, symm_n=symm_n, perm_table=perm_table, sym_rots=sym_rots),
                    msa, pair_for_str2str, R_in, T_in, xyz, state, idx, motif_mask, cyclic_reses
                )
        else:
            with profile_region("IterBlock.msa2msa"):
                msa = self.msa2msa(msa, pair, rbf_feat, state, symm_n=symm_n)
            with profile_region("IterBlock.msa2pair"):
                pair = self.msa2pair(msa, pair, symm_n=symm_n)
            with profile_region("IterBlock.pair2pair"):
                pair = self.pair2pair(pair, rbf_feat, block=block, t=t, symm_n=symm_n, perm_table=perm_table)
            # Hybrid adapters between pair and str2str (see _adapt_pair_for_str2str).
            pair_for_str2str = _adapt_pair_for_str2str(pair, self.symmetry_mode, symm_n, perm_table)
            with profile_region("IterBlock.str2str"):
                R, T, state, alpha = self.str2str(
                    msa, pair_for_str2str, R_in, T_in, xyz, state, idx,
                    motif_mask=motif_mask, cyclic_reses=cyclic_reses, top_k=0,
                    symm_n=symm_n, perm_table=perm_table, sym_rots=sym_rots
                )

        return msa, pair, R, T, state, alpha

class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 p_drop=0.15,
                 symmetry_mode=None):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block
        
        self.proj_state = nn.Linear(SE3_param_topk['l0_out_features'], SE3_param_full['l0_out_features'])
        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        p_drop=p_drop,
                                                        use_global_attn=True,
                                                        SE3_param=SE3_param_full,
                                                        symmetry_mode=symmetry_mode)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       p_drop=p_drop,
                                                       use_global_attn=False,
                                                       SE3_param=SE3_param_full,
                                                       symmetry_mode=symmetry_mode)
                                                       for i in range(n_main_block)])

        self.symmetry_mode = symmetry_mode
        self.proj_state2 = nn.Linear(SE3_param_full['l0_out_features'], SE3_param_topk['l0_out_features'])
        # Final SE(3) refinement
        if n_ref_block > 0:
            if symmetry_mode in ('restricted', 'hybrid_restr_str'):
                str2str_cls = RestrictedStr2Str
            else:
                # Includes None (null) and 'hybrid_restr_pair' — both want null Str2Str.
                str2str_cls = Str2Str
            self.str_refiner = str2str_cls(d_msa=d_msa, d_pair=d_pair,
                                           d_state=SE3_param_topk['l0_out_features'],
                                           SE3_param=SE3_param_topk,
                                           p_drop=p_drop)
    
        self.reset_parameter()
    def reset_parameter(self):
        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_state.bias)
        self.proj_state2 = init_lecun_normal(self.proj_state2)
        nn.init.zeros_(self.proj_state2.bias)

    def forward(self, seq, msa, msa_full, pair, xyz_in, state, idx, t=None, symm_n=1, perm_table=None, sym_rots=None, cyclic_reses=None, use_checkpoint=False, motif_mask=None):
        """
        input:
           seq: query sequence (B, L)
           msa: seed MSA embeddings (B, N, L, d_msa)
           msa_full: extra MSA embeddings (B, N, L, d_msa_full)
           pair: initial residue pair embeddings (B, L, L, d_pair)
           xyz_in: initial BB coordinates (B, L, n_atom, 3)
           state: initial state features containing mixture of query seq, sidechain, accuracy info (B, L, d_state)
           idx: residue index
           t: timestep
           motif_mask: bool tensor, True if motif position that is frozen, else False(L,) 
        """
        B, LR, LC = pair.shape[:3]

        if motif_mask is None:
            motif_mask = torch.zeros(LC).bool()

        R_in = torch.eye(3, device=xyz_in.device).reshape(1,1,3,3).expand(B, LC, -1, -1)
        T_in = xyz_in[:,:,1].clone()
        xyz_in = xyz_in - T_in.unsqueeze(-2)

        state = self.proj_state(state)

        R_s = list()
        T_s = list()
        alpha_s = list()

        # Saving instrumentation
        _save_pair = os.getenv("RFDIFF_SAVE_PAIR", "0") == "1"
        _save_se3 = os.getenv("RFDIFF_SAVE_SE3", "0") == "1"
        _pair_dir = os.getenv("RFDIFF_PAIR_DIR", "artifacts/pair_analysis")
        _pair_block_timesteps = {50, 40, 25, 10, 1}
        _se3_save_timesteps = {50, 25, 10, 1}
        _se3_save_blocks = {"main_0", "main_15", "main_31"}
        _t_val = t.item() if t is not None and hasattr(t, 'item') else t
        _d_num = int(os.getenv("RFDIFF_PAIR_DESIGN_NUM", "0"))

        def _save_pair_tensor(pair_tensor, filename):
            if not _save_pair:
                return
            d_dir = os.path.join(_pair_dir, f"design_{_d_num}")
            os.makedirs(d_dir, exist_ok=True)
            fpath = os.path.join(d_dir, filename)
            torch.save(pair_tensor.detach().cpu().half(), fpath)

        with profile_region("IterativeSimulator.extra_blocks"):
            for i_m in range(self.n_extra_block):
                R_in = R_in.detach()
                T_in = T_in.detach()
                xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
                block_name = f"extra_{i_m}"
                # Enable/disable SE3 saving for this specific block+timestep
                if _save_se3 and _t_val in _se3_save_timesteps and block_name in _se3_save_blocks:
                    os.environ["RFDIFF_SE3_BLOCK"] = block_name
                    os.environ["RFDIFF_SE3_TIMESTEP"] = str(_t_val)
                else:
                    os.environ.pop("RFDIFF_SAVE_SE3", None)
                msa_full, pair, R_in, T_in, state, alpha = self.extra_block[i_m](msa_full,
                                                                                 pair, R_in, T_in, xyz, state, idx,
                                                                                 motif_mask=motif_mask,
                                                                                 use_checkpoint=use_checkpoint,
                                                                                 cyclic_reses=cyclic_reses,
                                                                                 t=t, block=block_name,
                                                                                 symm_n=symm_n, perm_table=perm_table, sym_rots=sym_rots)
                if _t_val in _pair_block_timesteps:
                    _save_pair_tensor(pair, f"pair_t{_t_val}_block_{block_name}.pt")
                R_s.append(R_in)
                T_s.append(T_in)
                alpha_s.append(alpha)
                if _save_se3:
                    os.environ["RFDIFF_SAVE_SE3"] = "1"  # restore for next iteration

        with profile_region("IterativeSimulator.main_blocks"):
            for i_m in range(self.n_main_block):
                R_in = R_in.detach()
                T_in = T_in.detach()
                xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
                block_name = f"main_{i_m}"
                if _save_se3 and _t_val in _se3_save_timesteps and block_name in _se3_save_blocks:
                    os.environ["RFDIFF_SE3_BLOCK"] = block_name
                    os.environ["RFDIFF_SE3_TIMESTEP"] = str(_t_val)
                else:
                    os.environ.pop("RFDIFF_SAVE_SE3", None)
                msa, pair, R_in, T_in, state, alpha = self.main_block[i_m](msa,
                                                                           pair, R_in, T_in, xyz, state, idx,
                                                                           motif_mask=motif_mask,
                                                                           use_checkpoint=use_checkpoint,
                                                                           cyclic_reses=cyclic_reses,
                                                                           t=t, block=block_name,
                                                                           symm_n=symm_n, perm_table=perm_table, sym_rots=sym_rots)
                if _t_val in _pair_block_timesteps:
                    _save_pair_tensor(pair, f"pair_t{_t_val}_block_{block_name}.pt")
                R_s.append(R_in)
                T_s.append(T_in)
                alpha_s.append(alpha)
                if _save_se3:
                    os.environ["RFDIFF_SAVE_SE3"] = "1"

        _save_pair_tensor(pair, f"pair_t{_t_val}_final.pt")

        state = self.proj_state2(state)
        # if LR != LC:
        #     # Restricted mode: tile block-row to full for str_refiner
        #     pair = pair[:, :LR, ...]
        #     pair = symmetric_block_tile(pair, symm_n)

        with profile_region("IterativeSimulator.ref_blocks"):
            for i_m in range(self.n_ref_block):
                R_in = R_in.detach()
                T_in = T_in.detach()
                xyz = einsum('bnij,bnaj->bnai', R_in, xyz_in) + T_in.unsqueeze(-2)
                # Hybrid pair adapter: slice or tile depending on the mode.
                pair_for_refiner = _adapt_pair_for_str2str(pair, self.symmetry_mode, symm_n, perm_table)
                R_in, T_in, state, alpha = self.str_refiner(msa, pair_for_refiner, R_in, T_in, xyz, state, idx,
                                                            top_k=64, motif_mask=motif_mask,
                                                            cyclic_reses=cyclic_reses,
                                                            symm_n=symm_n, perm_table=perm_table, sym_rots=sym_rots)
                R_s.append(R_in)
                T_s.append(T_in)
                alpha_s.append(alpha)

        R_s = torch.stack(R_s, dim=0)
        T_s = torch.stack(T_s, dim=0)
        alpha_s = torch.stack(alpha_s, dim=0)

        return msa, pair, R_s, T_s, alpha_s, state
