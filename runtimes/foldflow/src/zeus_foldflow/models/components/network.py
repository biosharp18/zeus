""" Vectorfield network module.
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""
import functools as fn
import math

import torch
from zeus_foldflow.models.components import ipa_pytorch
from torch import nn
from torch.nn import functional as F

from zeus_foldflow.data import all_atom
from zeus_foldflow.data import utils as du
from zeus_foldflow.utils.symmetry_utils import get_symmetry_order


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Embedder(nn.Module):
    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        torch.set_default_dtype(torch.float32)
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
            get_timestep_embedding, embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding, embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res**2, -1])
        )

    def forward(
        self,
        *,
        seq_idx,
        t,
        fixed_mask,
        self_conditioning_ca,
        symm_n=1,
    ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.
            symm_n: cyclic / polyhedral group order. With ``symm_n == 1`` the
                block-diag trick is a no-op (provided ``seq_idx`` has unique
                entries) and this method behaves identically to the original
                FoldFlow embedder.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []
        assert num_res % symm_n == 0, (
            f"Number of residues ({num_res}) must be divisible by symmetry order ({symm_n})."
        )
        subunit_size = num_res // symm_n

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1)
        )
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]

        # Set inter-subunit offsets to large values to distinguish them from
        # intra-subunit offsets. Block-diagonally tile the intra-subunit offset
        # block, then mask the (now-zero) inter-subunit positions with the
        # sentinels +/-1000.  For symm_n == 1 this is a no-op as long as
        # ``seq_idx`` has unique entries (the only zeros sit on the main
        # diagonal, which is excluded by the strict triu/tril below).
        block = rel_seq_offset[:, :subunit_size, :subunit_size].squeeze(0)
        tiled_diag = torch.block_diag(*[block for _ in range(symm_n)])
        empty_space_mask = (tiled_diag == 0)
        is_upper = torch.triu(torch.ones_like(tiled_diag), diagonal=1).bool()
        is_lower = torch.tril(torch.ones_like(tiled_diag), diagonal=-1).bool()
        final_result = tiled_diag.masked_fill(empty_space_mask & is_upper, -1000)
        final_result = final_result.masked_fill(empty_space_mask & is_lower, 1000)
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
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class SymmetricEmbedder(Embedder):
    """Efficient embedder that only computes the embedding over the first
    subunit row (B, L_asu, N) instead of the full (B, N, N) — used when
    ``efficient_symmetry`` is True. Faithful port of FrameDiff's
    ``SymmetricEmbedder`` (model/score_network.py).
    """

    def _cross_concat_block_row(self, feats_1d, num_batch, num_res, subunit_size):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :subunit_size, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, subunit_size, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, subunit_size * num_res, -1])
        )

    def forward(
        self,
        *,
        seq_idx,
        t,
        fixed_mask,
        self_conditioning_ca,
        symm_n=1,
    ):
        """Embeds a set of inputs.

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
        assert num_res % symm_n == 0, (
            f"Number of residues ({num_res}) must be divisible by symmetry order ({symm_n})."
        )
        subunit_size = num_res // symm_n

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1)
        )
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [
            self._cross_concat_block_row(prot_t_embed, num_batch, num_res, subunit_size)
        ]
        # Positional index features.
        seq_idx = seq_idx.clone()
        seq_idx[:, subunit_size:] = 0  # we dont need the extra nodes anyways
        node_feats.append(self.index_embedder(seq_idx))

        # --- Efficient rel_seq_offset: O(L_asu * L) instead of O(L^2) ---
        # Intra-subunit block: real offsets [B, L_asu, L_asu]
        intra_offsets = seq_idx[:, :subunit_size, None] - seq_idx[:, None, :subunit_size]
        # Inter-subunit: all columns beyond first subunit are upper-triangle
        # (row < L_asu <= col), so they get the sentinel value -1000.
        inter_offsets = torch.full(
            (num_batch, subunit_size, num_res - subunit_size),
            -1000,
            device=seq_idx.device,
            dtype=intra_offsets.dtype,
        )
        rel_seq_offset_trunc = torch.cat(
            [intra_offsets, inter_offsets], dim=-1
        ).reshape([num_batch, subunit_size * num_res])

        pair_feats.append(self.index_embedder(rel_seq_offset_trunc))

        # --- Efficient distogram: O(L_asu * L) instead of O(L^2) ---
        if self._embed_conf.embed_self_conditioning:
            dists_2d = torch.linalg.norm(
                self_conditioning_ca[:, :subunit_size, None, :]
                - self_conditioning_ca[:, None, :, :],
                axis=-1,
            )[..., None]  # [B, L_asu, L, 1]
            lower = torch.linspace(
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
                device=self_conditioning_ca.device,
            )
            upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
            sc_dgram = ((dists_2d > lower) * (dists_2d < upper)).type(
                self_conditioning_ca.dtype
            )  # [B, L_asu, L, num_bins]
            pair_feats.append(
                sc_dgram.reshape([num_batch, subunit_size * num_res, -1])
            )

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, subunit_size, num_res, -1])

        return node_embed, edge_embed


class VectorFieldNetwork(nn.Module):
    def __init__(self, model_conf, flow_matcher):
        super(VectorFieldNetwork, self).__init__()
        self._model_conf = model_conf

        self._symmetry_type = model_conf.get("symmetry_type", "C")
        self.symm_n = get_symmetry_order(
            self._symmetry_type, model_conf.get("symmetry", 1)
        )
        self._efficient_symmetry = model_conf.get("efficient_symmetry", False)

        if self._efficient_symmetry:
            self.embedding_layer = SymmetricEmbedder(model_conf)
        else:
            self.embedding_layer = Embedder(model_conf)
        self.flow_matcher = flow_matcher
        self.vectorfield = ipa_pytorch.IpaNetwork(model_conf, flow_matcher)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats, use_context=False):
        """Forward computes the reverse conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats["res_mask"].type(torch.float32)  # [B, N]
        fixed_mask = input_feats["fixed_mask"].type(torch.float32)
        if self._efficient_symmetry:
            subunit_size = bb_mask.shape[1] // self.symm_n
            edge_mask = (
                bb_mask[:, :subunit_size, None] * bb_mask[..., None, :]
            )
        else:
            edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats["seq_idx"],
            t=input_feats["t"],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats["sc_ca_t"],
            symm_n=self.symm_n,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.vectorfield(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats["torsion_angles_sin_cos"][..., 2, :]
        psi_pred = self._apply_mask(model_out["psi"], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            "psi": psi_pred,
            "rot_vectorfield": model_out["rot_vectorfield"],
            "trans_vectorfield": model_out["trans_vectorfield"],
        }
        rigids_pred = model_out["final_rigids"]
        pred_out["rigids"] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out["atom37"] = bb_representations[0].to(rigids_pred.device)
        pred_out["atom14"] = bb_representations[-1].to(rigids_pred.device)
        return pred_out
