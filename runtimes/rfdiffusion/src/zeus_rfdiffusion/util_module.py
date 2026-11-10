import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import copy
import dgl
from zeus_rfdiffusion.util import base_indices, RTs_by_torsion, xyzs_in_base_frame, rigid_from_3_points


def find_breaks(ix, thresh=35):
    # finds positions in ix where the jump is greater than 100
    breaks = np.where(np.diff(ix) > thresh)[0]
    return np.array(breaks)+1


def init_lecun_normal(module):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape):
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

def init_lecun_normal_param(weight):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape):
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
    return weight

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def get_seqsep(idx, cyclic=None):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    seqsep = idx[:,None,:] - idx[:,:,None]
    sign = torch.sign(seqsep)
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0 # if bonded -- 1.0 / else 0.0
    neigh = sign * neigh

    # add cyclic edges
    breaks = find_breaks(idx.squeeze().cpu().numpy())
    chainids = np.zeros_like(idx.squeeze().cpu().numpy())
    for i, b in enumerate(breaks):
        chainids[b:] = i+1
    chainids = torch.from_numpy(chainids).to(device=idx.device)

    # add cyclic edges with multiple chains
    if (cyclic is not None):
        for chid in torch.unique(chainids):
            is_chid = chainids==chid
            cur_cyclic = cyclic*is_chid
            cur_cres = cur_cyclic.nonzero()

            if cur_cyclic.sum()>=2:
                neigh[:,cur_cres[-1],cur_cres[0]] = 1
                neigh[:,cur_cres[0],cur_cres[-1]] = -1

    return neigh.unsqueeze(-1)

import time as _time
import os as _os
from contextlib import contextmanager
from collections import defaultdict

# Lightweight profiling: tracks wall time + peak GPU memory per named region.
# Gated by RFDIFF_PROFILE=1. When off, the context manager is a no-op.
_PROFILE_ENABLED = _os.getenv("RFDIFF_PROFILE", "0") == "1"
_PROFILE_DATA = defaultdict(lambda: {"count": 0, "time_sum": 0.0, "peak_mb_max": 0.0})

@contextmanager
def profile_region(name):
    """Track wall time and peak GPU memory for a named code region.

    Enable with RFDIFF_PROFILE=1. Results collected in _PROFILE_DATA and
    printed via print_profile_summary().
    """
    if not _PROFILE_ENABLED:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = _time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak = 0.0
    elapsed = _time.perf_counter() - start
    entry = _PROFILE_DATA[name]
    entry["count"] += 1
    entry["time_sum"] += elapsed
    entry["peak_mb_max"] = max(entry["peak_mb_max"], peak)

def print_profile_summary():
    """Print collected profile_region stats and reset."""
    if not _PROFILE_DATA:
        print("[profile] No data collected. Set RFDIFF_PROFILE=1 to enable.")
        return
    print(f"\n{'Region':<50} {'Calls':>6} {'Total(s)':>9} {'Avg(ms)':>9} {'Peak(MB)':>10}")
    print("-" * 86)
    for name, d in sorted(_PROFILE_DATA.items(), key=lambda x: -x[1]["time_sum"]):
        avg_ms = d["time_sum"] / d["count"] * 1000
        print(f"{name:<50} {d['count']:>6} {d['time_sum']:>9.3f} {avg_ms:>9.2f} {d['peak_mb_max']:>10.1f}")
    print()
    _PROFILE_DATA.clear()


def symmetric_block_tile(x, perm_table):
    """Tile a block-row (B, LR, LC, D) into a full pair matrix (B, LC, LC, D).

    Uses a group permutation table to permute block-columns for each block-row.
    For cyclic groups the table is circulant and this is equivalent to the old
    torch.roll approach. For polyhedral groups each row is a different
    permutation.

    Args:
        x:          (B, LR, LC, D...) — restricted pair features (block-row 0).
        perm_table: (S, S) LongTensor where entry [g, h] = index of g⁻¹h in
                    the group. S = symm_n, LR = LC // S.
                    Also accepts a plain int (symm_n) for backward compatibility,
                    in which case a cyclic permutation table is built on the fly.
    """
    if isinstance(perm_table, int):
        # Backward compat: build cyclic perm table from integer symm_n.
        S = perm_table
        perm_table = torch.stack([torch.roll(torch.arange(S), shifts=g) for g in range(S)])
    S = perm_table.shape[0]
    B, LR, LC = x.shape[:3]
    rows = []
    for g in range(S):
        perm = perm_table[g]  # (S,)
        col_idx = (perm[:, None] * LR + torch.arange(LR, device=x.device)[None, :]).reshape(-1)
        rows.append(x[:, :, col_idx])
    return torch.cat(rows, dim=1)


def build_group_permutations(sym_rots):
    """Build the g⁻¹h permutation table for a finite rotation group.

    Given S rotation matrices [R_0, R_1, ..., R_{S-1}] (with R_0 = identity),
    returns an (S, S) integer tensor where entry [g, h] is the index k such
    that R_k = R_g⁻¹ @ R_h.

    For cyclic groups the table is circulant (every row is a roll of row 0).
    For polyhedral groups each row is a different permutation.

    Also returns diagnostic info: the margin between the best and second-best
    match for each (g, h) pair, so the caller can assess numerical reliability.

    Args:
        sym_rots: list of S rotation matrices, each (3, 3). Can be torch
                  tensors or numpy arrays. Identity must be at index 0.

    Returns:
        perm_table: (S, S) LongTensor — perm_table[g, h] = index of R_g⁻¹ R_h.
        margins:    (S, S) float tensor — Frobenius-norm gap between the best
                    and second-best match for each (g, h).
    """

    # Normalize to numpy float64 for precision.
    rots = np.stack([
        r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        for r in sym_rots
    ]).astype(np.float64)  # (S, 3, 3)
    S = len(rots)

    perm_table = np.zeros((S, S), dtype=np.int64)
    margins = np.zeros((S, S), dtype=np.float64)

    for g in range(S):
        Rg_inv = rots[g].T  # inverse = transpose for rotation matrices
        for h in range(S):
            product = Rg_inv @ rots[h]  # R_g⁻¹ R_h
            diffs = np.linalg.norm(rots - product[None], axis=(1, 2))  # (S,)
            sorted_idx = np.argsort(diffs)
            perm_table[g, h] = sorted_idx[0]
            margins[g, h] = diffs[sorted_idx[1]] - diffs[sorted_idx[0]]

    return torch.from_numpy(perm_table).long(), torch.from_numpy(margins).float()


def make_asu_full_graph(xyz, pair_restricted, idx, L_asu, kmin=9, perm_table=None):
    """Build a graph with edges pointing only to ASU nodes (first L_asu).

    Sources can be any of the L nodes. This reduces edge count from L*(L-1) to
    L_asu*(L-1), a factor of symm_n reduction.

    Args:
        xyz: CA coordinates (B, L, 3)
        pair_restricted: pair features (B, L_asu, L, E) — restricted to ASU rows
        idx: residue indices (B, L)
        L_asu: number of residues in the asymmetric unit
        perm_table: (S, S) LongTensor — maps (src_block, dst_block) to canonical
                    block index in block-row 0. If None, falls back to cyclic.
    Returns:
        G: DGL graph with B*L nodes but only edges to ASU destinations
        edge_feats: (num_edges, E, 1)
    """
    B, L = xyz.shape[:2]
    device = xyz.device

    # sep[b, i, j] where i in [0, L_asu) = destination, j in [0, L) = source
    sep = idx[:, None, :] - idx[:, :L_asu, None]  # (B, L_asu, L)
    b, i, j = torch.where(sep.abs() > 0)  # i in [0, L_asu), j in [0, L)

    src = b * L + j   # source: any node
    tgt = b * L + i   # destination: ASU nodes only
    G = dgl.graph((src, tgt), num_nodes=B * L).to(device)
    # rel_pos convention: xyz[tgt] - xyz[src] = xyz[i] - xyz[j] (matches make_full_graph)
    G.edata['rel_pos'] = (xyz[b, i, :] - xyz[b, j, :]).detach()

    # Look up pair_full[src=j, tgt=i] from the restricted pair (block-row 0).
    # pair_full[g_s*L_asu + a, g_t*L_asu + b] = pair_restricted[a, perm[g_s, g_t]*L_asu + b]
    # Here tgt i < L_asu so g_t = 0, local_tgt = i.
    block_k = j // L_asu       # source block
    local_j = j % L_asu        # source local index
    if perm_table is not None:
        perm = perm_table.to(device)
        inv_block = perm[block_k, 0]  # perm[g_src, g_tgt=0]
        col_shifted = inv_block * L_asu + i
    else:
        # Cyclic fallback: matches original formula exactly for circulant perm_table.
        col_shifted = (i - block_k * L_asu) % L
    edge_feats = pair_restricted[b, local_j, col_shifted][..., None]

    return G, edge_feats


def make_full_graph_from_restricted(xyz, pair_restricted, idx, L_asu, perm_table=None):
    """Build a full L²-edge graph but index edge features from restricted pair (L_asu, L).

    For cyclic groups (or perm_table=None) uses circulant indexing.
    For polyhedral groups pass the group's perm_table.

    Args:
        xyz: CA coordinates (B, L, 3)
        pair_restricted: pair features (B, L_asu, L, E) — restricted to ASU rows
        idx: residue indices (B, L)
        L_asu: number of residues in the asymmetric unit
        perm_table: (S, S) LongTensor. If None, falls back to cyclic.
    Returns:
        G: DGL graph with all L² edges (same as make_full_graph)
        edge_feats: (num_edges, E, 1)
    """
    B, L = xyz.shape[:2]
    device = xyz.device
    LC = pair_restricted.shape[2]

    sep = idx[:, None, :] - idx[:, :, None]  # (B, L, L)
    b, i, j = torch.where(sep.abs() > 0)

    src = b * L + i
    tgt = b * L + j
    G = dgl.graph((src, tgt), num_nodes=B * L).to(device)
    G.edata['rel_pos'] = (xyz[b, j, :] - xyz[b, i, :]).detach()

    # Look up pair_full[src=i, tgt=j] from restricted pair.
    # i = block_src * L_asu + local_src, j = block_tgt * L_asu + local_tgt
    # pair_full[i, j] = pair_restricted[local_src, perm[block_src, block_tgt]*L_asu + local_tgt]
    block_src = torch.div(i, L_asu, rounding_mode='trunc')
    local_src = i % L_asu
    block_tgt = torch.div(j, L_asu, rounding_mode='trunc')
    local_tgt = j % L_asu
    if perm_table is not None:
        perm = perm_table.to(device)
        col_shifted = perm[block_src, block_tgt] * L_asu + local_tgt
    else:
        col_shifted = (j - block_src * L_asu) % LC

    edge_feats = pair_restricted[b, local_src, col_shifted][..., None]
    return G, edge_feats


def compute_wigner_d2(angle, device='cpu'):
    """Compute the degree-2 Wigner D-matrix for a z-axis rotation.

    Uses a one-shot equivariance test on a tiny SE3Transformer to extract
    the D2 matrix in the correct convention. The result depends only on the
    rotation angle, not on model weights or input data.

    Args:
        angle: rotation angle in radians (float)
        device: torch device
    Returns:
        D2: (5, 5) rotation matrix for degree-2 features
    """
    from se3_transformer.model import SE3Transformer
    from se3_transformer.model.fiber import Fiber
    from se3_transformer.model.basis import get_basis, update_basis_with_fused
    from se3_transformer.model.transformer import get_populated_edge_features

    L_test = 5
    src, tgt = [], []
    for i in range(L_test):
        for j in range(L_test):
            if i != j:
                src.append(j)
                tgt.append(i)

    xyz_test = torch.randn(L_test, 3, device=device)
    G = dgl.graph((src, tgt), num_nodes=L_test).to(device)
    G.edata['rel_pos'] = (xyz_test[tgt] - xyz_test[src]).detach()

    c, s = float(np.cos(angle)), float(np.sin(angle))
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32, device=device)

    xyz_rot = xyz_test @ R.T
    G_rot = dgl.graph((src, tgt), num_nodes=L_test).to(device)
    G_rot.edata['rel_pos'] = (xyz_rot[tgt] - xyz_rot[src]).detach()

    # Tiny SE3 just to get the degree-2 representation
    se3_test = SE3Transformer(
        num_layers=1, fiber_in=Fiber({0: 4, 1: 3}),
        fiber_hidden=Fiber.create(3, 8), fiber_out=Fiber({0: 4, 1: 2}),
        num_heads=2, channels_div=2, fiber_edge=Fiber({0: 4}), use_layer_norm=True
    ).to(device).eval()

    attn_block = se3_test.graph_modules[0]
    nf = {'0': torch.randn(L_test, 4, 1, device=device), '1': torch.randn(L_test, 3, 3, device=device)}
    nf_rot = {'0': nf['0'].clone(), '1': nf['1'] @ R.T}
    ef = {'0': torch.randn(G.num_edges(), 4, 1, device=device)}

    ep = get_populated_edge_features(G.edata['rel_pos'], ef)
    ep_rot = get_populated_edge_features(G_rot.edata['rel_pos'], ef)
    basis = get_basis(G.edata['rel_pos'], max_degree=se3_test.max_degree, compute_gradients=False,
                      use_pad_trick=False, amp=False)
    basis = update_basis_with_fused(basis, se3_test.max_degree, use_pad_trick=False, fully_fused=False)
    basis_rot = get_basis(G_rot.edata['rel_pos'], max_degree=se3_test.max_degree, compute_gradients=False,
                          use_pad_trick=False, amp=False)
    basis_rot = update_basis_with_fused(basis_rot, se3_test.max_degree, use_pad_trick=False, fully_fused=False)

    with torch.no_grad():
        out = attn_block(nf, ep, graph=G, basis=basis)
        out_rot = attn_block(nf_rot, ep_rot, graph=G_rot, basis=basis_rot)

    f2 = out['2'].reshape(-1, 5).double()
    f2_rot = out_rot['2'].reshape(-1, 5).double()
    D2_T = torch.linalg.lstsq(f2, f2_rot).solution.float()
    return D2_T.T.to(device)


def make_full_graph(xyz, pair, idx, top_k=64, kmin=9):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    b,i,j = torch.where(sep.abs() > 0)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None] #what to do for edge feats of collapsed nodes? they are the same

def make_topk_graph(xyz, pair, idx, top_k=64, kmin=32, eps=1e-6):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # distance map from current CA coordinates
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    D = D + sep*eps
    
    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = torch.where(cond)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None]

def make_asu_dest_topk_graph(xyz, pair_restricted, idx, L_asu, top_k=64, kmin=32, eps=1e-6, perm_table=None):
    """Build top-k graph where only ASU nodes (0..L_asu-1) are destinations.

    All L nodes exist (needed as message sources), but edges only go TO ASU.
    Uses two-way top-k to capture edges from both perspectives:
      1) For each ASU node j, find top-k nearest among all L → edges (i→j)
      2) For each node i, find top-k nearest among ASU → edges (i→j_asu)
    Plus sequential proximity edges to ASU nodes.

    Distance computation is O(L * L_asu) instead of O(L²).
    Edge features use circulant indexing into the restricted pair.
    """
    B, L = xyz.shape[:2]
    device = xyz.device
    LC = pair_restricted.shape[2]

    # Distances: ASU rows to all columns — (B, L_asu, L)
    D = torch.cdist(xyz[:, :L_asu], xyz)
    D[:, range(L_asu), range(L_asu)] += 999.9  # mask self-loops

    # Seq sep for (ASU dest j, any src i): (B, L_asu, L)
    sep = (idx[:, :L_asu].unsqueeze(2) - idx.unsqueeze(1)).abs().float()
    sep[:, range(L_asu), range(L_asu)] += 999.9
    D_biased = D + sep * eps

    # Direction 1: per-ASU-node, top-k from all L sources
    k1 = min(top_k, L - 1)
    _, topk_1 = torch.topk(D_biased, k1, largest=False, dim=2)  # (B, L_asu, k1)
    cond = torch.zeros(B, L_asu, L, device=device)
    cond.scatter_(2, topk_1, 1.0)

    # Direction 2: per-all-node, top-k among ASU destinations
    D_t = D_biased.transpose(1, 2)  # (B, L, L_asu)
    k2 = min(top_k, L_asu - 1)
    _, topk_2 = torch.topk(D_t, k2, largest=False, dim=2)  # (B, L, k2)
    # topk_2[b, i, :] = ASU node indices that are closest to node i
    # → edges (i → topk_2[b,i,:])  — mark in (B, L_asu, L) as cond_2[b, j, i] = 1
    cond_2 = torch.zeros(B, L_asu, L, device=device)
    b_exp = torch.arange(B, device=device)[:, None, None].expand_as(topk_2)
    i_exp = torch.arange(L, device=device)[None, :, None].expand_as(topk_2)
    cond_2[b_exp.reshape(-1), topk_2.reshape(-1), i_exp.reshape(-1)] = 1.0

    # Union both directions + sequential proximity
    final_cond = ((cond + cond_2) > 0) | (sep < kmin)
    b_idx, j_idx, i_idx = torch.where(final_cond)

    src = b_idx * L + i_idx
    tgt = b_idx * L + j_idx
    G = dgl.graph((src, tgt), num_nodes=B * L).to(device)
    G.edata['rel_pos'] = (xyz[b_idx, j_idx] - xyz[b_idx, i_idx]).detach()

    # Edge features: look up pair_full[src=i, dst=j] from restricted pair.
    # Source node i is in block block_k = i // L_asu, local index local_i.
    # Destination j is in ASU (j < L_asu), so dst block = 0, local dst = j.
    # pair_full[g_s*L_asu+a, 0*L_asu+b] = pair_restricted[a, perm[g_s, 0]*L_asu + b]
    block_k = torch.div(i_idx, L_asu, rounding_mode='trunc')
    local_i = i_idx % L_asu
    if perm_table is not None:
        perm = perm_table.to(device)
        inv_block = perm[block_k, 0]
        col_shifted = inv_block * L_asu + j_idx
    else:
        # Cyclic fallback
        col_shifted = (j_idx - block_k * L_asu) % LC
    edge_feats = pair_restricted[b_idx, local_i, col_shifted][..., None]

    return G, edge_feats


def make_asu_dest_topk_graph_matched(xyz, pair_restricted, idx, L_asu,
                                     top_k=64, kmin=32, eps=1e-6, perm_table=None):
    """Top-k graph with ASU destinations, matching `make_topk_graph`'s edge
    rule (source picks top-k from full L, OR sep < kmin).

    This replaces `make_asu_dest_topk_graph`, which used a DIFFERENT edge
    rule (two-way top-k with restricted candidate pools) that produced more
    edges than `make_topk_graph` and put the SE3 in an OOD regime. This
    variant produces the same edges `make_topk_graph` would, filtered to
    ASU destinations.

    Memory budget O(L_asu * L): no full L x L matrices are materialized.
    The key trick is symmetry — for source i in subunit g, local `a`:
        D[i, j] = D[a, perm[g, h(j)] * L_asu + b(j)]
    So source i's top-k is a permutation of source a's top-k. We compute
    top-k for ASU sources only (L_asu rows), then derive top-k ASU
    destinations for all S*L_asu sources by group-action permutation.
    """
    B, L = xyz.shape[:2]
    device = xyz.device
    LC = pair_restricted.shape[2]
    S = L // L_asu

    if perm_table is None:
        # Cyclic fallback
        perm = torch.stack([torch.roll(torch.arange(S), shifts=g) for g in range(S)]).to(device)
    else:
        perm = perm_table.to(device)

    # --- O(L_asu * L) distances from ASU sources to ALL destinations ---
    D = torch.cdist(xyz[:, :L_asu], xyz)  # (B, L_asu, L)
    # Mask ASU-to-self entries (self-loop)
    D[:, range(L_asu), range(L_asu)] += 999.9

    sep_asu = (idx[:, :L_asu].unsqueeze(2) - idx.unsqueeze(1)).abs().float()  # (B, L_asu, L)
    sep_asu[:, range(L_asu), range(L_asu)] += 999.9
    D_biased = D + sep_asu * eps

    # Top-k over full-L destinations, per ASU source: same as what
    # make_topk_graph computes for rows a < L_asu.
    k = min(top_k, L - 1)
    _, E_idx = torch.topk(D_biased, k, largest=False, dim=2)  # (B, L_asu, k)

    # --- Build (B, L, L_asu) bool mask of top-k edges targeting ASU ---
    # Edge (src=i → tgt=j) with j < L_asu is in i's top-k iff source `a`'s
    # top-k (where a = i % L_asu, g = i // L_asu) contains an entry at
    # block perm[g, 0] with local index j.
    topk_cond = torch.zeros(B, L, L_asu, dtype=torch.bool, device=device)
    for g in range(S):
        target_block = int(perm[g, 0].item())  # = inv(g)
        lo = target_block * L_asu
        hi = lo + L_asu
        match = (E_idx >= lo) & (E_idx < hi)  # (B, L_asu, k)
        b_idx, a_idx, k_idx = torch.where(match)
        j_local = E_idx[b_idx, a_idx, k_idx] - lo  # (N,) in [0, L_asu)
        src_i = g * L_asu + a_idx                  # source index in [0, L)
        topk_cond[b_idx, src_i, j_local] = True

    # --- Sequential proximity: sep < kmin for (any source, ASU destination) ---
    sep_sources = (idx.unsqueeze(2) - idx[:, :L_asu].unsqueeze(1)).abs().float()  # (B, L, L_asu)
    # Mask self-loops (source i == ASU destination i for i < L_asu)
    diag_idx = torch.arange(L_asu, device=device)
    sep_sources[:, diag_idx, diag_idx] += 999.9
    prox_cond = sep_sources < kmin

    # Union top-k and proximity edges
    edge_cond = topk_cond | prox_cond  # (B, L, L_asu), bool
    b_idx, i_idx, j_idx = torch.where(edge_cond)

    src = b_idx * L + i_idx
    tgt = b_idx * L + j_idx
    G = dgl.graph((src, tgt), num_nodes=B * L).to(device)
    G.edata['rel_pos'] = (xyz[b_idx, j_idx] - xyz[b_idx, i_idx]).detach()

    # Edge features via perm_table lookup (same convention as other ASU-dest builders)
    block_k = torch.div(i_idx, L_asu, rounding_mode='trunc')
    local_i = i_idx % L_asu
    if perm_table is not None:
        inv_block = perm[block_k, 0]
        col_shifted = inv_block * L_asu + j_idx
    else:
        col_shifted = (j_idx - block_k * L_asu) % LC
    edge_feats = pair_restricted[b_idx, local_i, col_shifted][..., None]

    return G, edge_feats


def make_asu_dest_full_graph(xyz, pair_restricted, idx, L_asu, perm_table=None):
    """Build full graph where only ASU nodes are destinations.

    All edges (i→j) with j < L_asu, i != j. O(L * L_asu) edges.
    Edge features use circulant indexing into the restricted pair.
    """
    B, L = xyz.shape[:2]
    device = xyz.device
    LC = pair_restricted.shape[2]

    sep = idx[:, :L_asu].unsqueeze(2) - idx.unsqueeze(1)  # (B, L_asu, L)
    b_idx, j_idx, i_idx = torch.where(sep.abs() > 0)

    src = b_idx * L + i_idx
    tgt = b_idx * L + j_idx
    G = dgl.graph((src, tgt), num_nodes=B * L).to(device)
    G.edata['rel_pos'] = (xyz[b_idx, j_idx] - xyz[b_idx, i_idx]).detach()

    block_k = torch.div(i_idx, L_asu, rounding_mode='trunc')
    local_i = i_idx % L_asu
    if perm_table is not None:
        perm = perm_table.to(device)
        inv_block = perm[block_k, 0]
        col_shifted = inv_block * L_asu + j_idx
    else:
        # Cyclic fallback
        col_shifted = (j_idx - block_k * L_asu) % LC
    edge_feats = pair_restricted[b_idx, local_i, col_shifted][..., None]

    return G, edge_feats


def make_rotX(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,1,1] = angs[:,:,0]/NORM
    RTs[:,:,1,2] = -angs[:,:,1]/NORM
    RTs[:,:,2,1] = angs[:,:,1]/NORM
    RTs[:,:,2,2] = angs[:,:,0]/NORM
    return RTs

# rotate about the z axis
def make_rotZ(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,0,0] = angs[:,:,0]/NORM
    RTs[:,:,0,1] = -angs[:,:,1]/NORM
    RTs[:,:,1,0] = angs[:,:,1]/NORM
    RTs[:,:,1,1] = angs[:,:,0]/NORM
    return RTs

# rotate about an arbitrary axis
def make_rot_axis(angs, u, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    ct = angs[:,:,0]/NORM
    st = angs[:,:,1]/NORM
    u0 = u[:,:,0]
    u1 = u[:,:,1]
    u2 = u[:,:,2]

    RTs[:,:,0,0] = ct+u0*u0*(1-ct)
    RTs[:,:,0,1] = u0*u1*(1-ct)-u2*st
    RTs[:,:,0,2] = u0*u2*(1-ct)+u1*st
    RTs[:,:,1,0] = u0*u1*(1-ct)+u2*st
    RTs[:,:,1,1] = ct+u1*u1*(1-ct)
    RTs[:,:,1,2] = u1*u2*(1-ct)-u0*st
    RTs[:,:,2,0] = u0*u2*(1-ct)-u1*st
    RTs[:,:,2,1] = u1*u2*(1-ct)+u0*st
    RTs[:,:,2,2] = ct+u2*u2*(1-ct)
    return RTs

class ComputeAllAtomCoords(nn.Module):
    def __init__(self):
        super(ComputeAllAtomCoords, self).__init__()

        self.base_indices = nn.Parameter(base_indices, requires_grad=False)
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False)

    def forward(self, seq, xyz, alphas, non_ideal=False, use_H=True):
        B,L = xyz.shape[:2]

        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], non_ideal=non_ideal)

        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)

        # bb
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq,0,:], make_rotX(alphas[:,:,0,:]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,1,:], make_rotX(alphas[:,:,1,:]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,2,:], make_rotX(alphas[:,:,2,:]))

        # CB bend
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5*(basexyzs[:,:,2,:3]+basexyzs[:,:,0,:3])
        CAr = (basexyzs[:,:,1,:3])
        CBr = (basexyzs[:,:,4,:3])
        CBrotaxis1 = (CBr-CAr).cross(NCr-CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True)+1e-8
        
        # CB twist
        NCp = basexyzs[:,:,2,:3] - basexyzs[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCr, dim=-1, keepdim=True)/ torch.sum(NCr*NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr-CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True)+1e-8
        
        CBrot1 = make_rot_axis(alphas[:,:,7,:], CBrotaxis1 )
        CBrot2 = make_rot_axis(alphas[:,:,8,:], CBrotaxis2 )
        
        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, CBrot1,CBrot2)
        
        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim', 
            RTF8, 
            self.RTs_in_base_frame[seq,3,:], 
            make_rotX(alphas[:,:,3,:]), 
            make_rotZ(alphas[:,:,9,:]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF4, self.RTs_in_base_frame[seq,4,:],make_rotX(alphas[:,:,4,:]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF5,self.RTs_in_base_frame[seq,5,:],make_rotX(alphas[:,:,5,:]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF6,self.RTs_in_base_frame[seq,6,:],make_rotX(alphas[:,:,6,:]))

        RTframes = torch.stack((
            RTF0,RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7,RTF8
        ),dim=2)

        xyzs = torch.einsum(
            'brtij,brtj->brti', 
            RTframes.gather(2,self.base_indices[seq][...,None,None].repeat(1,1,1,4,4)), basexyzs
        )

        if use_H:
            return RTframes, xyzs[...,:3]
        else:
            return RTframes, xyzs[...,:14,:3]
