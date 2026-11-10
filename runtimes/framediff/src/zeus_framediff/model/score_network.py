"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from zeus_framediff.data import utils as du
from zeus_framediff.data import all_atom
from zeus_framediff.model import ipa_pytorch
from zeus_framediff.data.symmetry_utils import get_symmetry_order
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def _chain_masked_intra_block(intra_offsets, chain_sizes, sentinel: int = 1000):
    """Mask inter-chain pairs within a subunit's intra-offset block with ±sentinel.

    Generalizes the sentinel trick already used for inter-subunit pairs
    (see Embedder/SymmetricEmbedder). Positions where residues i and j
    belong to different sub-chains of the same subunit get replaced with
    -sentinel (upper triangle) or +sentinel (lower triangle), matching
    the existing convention so downstream index_embedder sees the same
    "very distant in sequence" signal it already sees for inter-subunit
    pairs.

    Args:
        intra_offsets: [B, L_asu, L_asu] raw seq_idx[i] - seq_idx[j].
        chain_sizes: list of sub-chain lengths (must sum to L_asu).
        sentinel: magnitude of the sentinel (default 1000, same as
            existing inter-subunit sentinel).

    Returns:
        [B, L_asu, L_asu] with inter-chain positions overwritten.
    """
    if chain_sizes is None or len(chain_sizes) <= 1:
        return intra_offsets
    L_asu = intra_offsets.shape[-1]
    assert sum(chain_sizes) == L_asu, \
        f"chain_sizes {list(chain_sizes)} must sum to L_asu={L_asu}"
    device = intra_offsets.device
    chain_id = torch.zeros(L_asu, dtype=torch.long, device=device)
    start = 0
    for c, size in enumerate(chain_sizes):
        chain_id[start:start + size] = c
        start += size
    inter_chain = (chain_id[:, None] != chain_id[None, :])  # [L_asu, L_asu]
    upper = torch.triu(torch.ones_like(inter_chain, dtype=torch.bool), diagonal=1)
    lower = torch.tril(torch.ones_like(inter_chain, dtype=torch.bool), diagonal=-1)
    out = intra_offsets.clone()
    out = out.masked_fill((inter_chain & upper)[None], -sentinel)
    out = out.masked_fill((inter_chain & lower)[None], +sentinel)
    return out


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
            symm_n=None
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []
        assert num_res % symm_n == 0, f"Number of residues ({num_res}) must be divisible by symmetry order ({symm_n})."
        subunit_size = num_res // symm_n

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]

        # Set inter-subunit offsets to large values to distinguish them from intra-subunit offsets.
        # Create subunit index for each residue
        block = rel_seq_offset[:,:subunit_size, :subunit_size].squeeze(0)
        # Multi-chain subunits: mask inter-chain pairs within the subunit
        # block BEFORE tiling. block_diag then produces a chain-level
        # block-diagonal (G*C blocks) with sentinels off that diagonal.
        chain_sizes = self._model_conf.get("subunit_chain_sizes", None)
        block = _chain_masked_intra_block(block[None], chain_sizes).squeeze(0)
        tiled_diag = torch.block_diag(*[block for _ in range(symm_n)])
        empty_space_mask = (tiled_diag == 0)
        is_upper = torch.triu(torch.ones_like(tiled_diag), diagonal=1).bool()
        is_lower = torch.tril(torch.ones_like(tiled_diag), diagonal=-1).bool()
        final_result = tiled_diag.masked_fill(empty_space_mask & is_upper, -1000)
        final_result = final_result.masked_fill(empty_space_mask & is_lower, 1000)
        #Diagonally tile rel_seq_offset and set inter-subunit offsets to large values to distinguish them from intra-subunit offsets.
        rel_seq_offset = final_result.unsqueeze(0)
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))
        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float()) # [1,10000,66], [1,10000,32], [1,10000,22]
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed

class SymmetricEmbedder(Embedder):
    def _cross_concat_block_row(self, feats_1d, num_batch, num_res, subunit_size):
        return torch.cat([
            torch.tile(feats_1d[:, :subunit_size, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, subunit_size, 1, 1)),
        ], dim=-1).float().reshape([num_batch, subunit_size * num_res, -1])

    def forward(
        self,
        *,
        seq_idx,
        t,
        fixed_mask,
        self_conditioning_ca,
        symm_n=None
    ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, L_asu, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []
        assert num_res % symm_n == 0, f"Number of residues ({num_res}) must be divisible by symmetry order ({symm_n})."
        subunit_size = num_res // symm_n

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat_block_row(
            prot_t_embed, num_batch, num_res, subunit_size)]
        # Positional index features.
        seq_idx[:,subunit_size:] = 0 #we dont need the extra nodes anyways
        node_feats.append(self.index_embedder(seq_idx))

        # --- Efficient rel_seq_offset: O(L_asu * L) instead of O(L^2) ---
        # Intra-subunit block: real offsets [B, L_asu, L_asu]
        intra_offsets = seq_idx[:, :subunit_size, None] - seq_idx[:, None, :subunit_size]
        # Multi-chain subunits: mask inter-chain pairs within the subunit
        # with sentinel values so they look "sequence-distant" to the
        # pair embedder, matching how inter-subunit pairs are handled.
        chain_sizes = self._model_conf.get("subunit_chain_sizes", None)
        intra_offsets = _chain_masked_intra_block(intra_offsets, chain_sizes)
        # Inter-subunit: all columns beyond first subunit are upper-triangle
        # (row < L_asu <= col), so they get the sentinel value -1000.
        inter_offsets = torch.full(
            (num_batch, subunit_size, num_res - subunit_size),
            -1000, device=seq_idx.device, dtype=intra_offsets.dtype)
        rel_seq_offset_trunc = torch.cat(
            [intra_offsets, inter_offsets], dim=-1
        ).reshape([num_batch, subunit_size * num_res])

        pair_feats.append(self.index_embedder(rel_seq_offset_trunc))

        # --- Efficient distogram: O(L_asu * L) instead of O(L^2) ---
        if self._embed_conf.embed_self_conditioning:
            dists_2d = torch.linalg.norm(
                self_conditioning_ca[:, :subunit_size, None, :] -
                self_conditioning_ca[:, None, :, :],
                axis=-1,
            )[..., None]  # [B, L_asu, L, 1]
            lower = torch.linspace(
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
                device=self_conditioning_ca.device)
            upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
            sc_dgram = ((dists_2d > lower) * (dists_2d < upper)).type(
                self_conditioning_ca.dtype)  # [B, L_asu, L, num_bins]
            pair_feats.append(
                sc_dgram.reshape([num_batch, subunit_size * num_res, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, subunit_size, num_res, -1])

        return node_embed, edge_embed
    
class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        if self._model_conf.efficient_symmetry:
            self.embedding_layer = SymmetricEmbedder(model_conf)
        else:
            self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)

        symm_n = get_symmetry_order(
            self._model_conf.get("symmetry_type", "C"),
            self._model_conf.symmetry,
            self._model_conf.get("inner_symmetry_type", None),
            self._model_conf.get("inner_symmetry", None),
        )
        if self._model_conf.efficient_symmetry:
            subunit_size = bb_mask.shape[1] // symm_n
            edge_mask = (
                bb_mask[:, :subunit_size, None] * bb_mask[..., None, :]
            )
        else:
            edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        #Set node embeddings to be init the same
        # Initial embeddings of positonal and relative indices.
        #input_feats["seq_idx"][:,50:] = input_feats["seq_idx"][:,:50]

        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
            symm_n=symm_n,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])
        pred_out = {
            'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out['atom37'] = bb_representations[0].to(rigids_pred.device)
        pred_out['atom14'] = bb_representations[-1].to(rigids_pred.device)
        return pred_out
