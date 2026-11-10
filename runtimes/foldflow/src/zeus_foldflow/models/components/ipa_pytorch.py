"""Fork of Openfold's IPA."""

import math
import os
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from zeus_foldflow.models.components.positional_embeddings import PositionalEmbedding
from scipy.stats import truncnorm


def _save_attn_tensor(attn, block_idx, step_idx):
    """Save post-softmax IPA attention for analysis.

    Controlled by env vars:
      FOLDFLOW_SAVE_ATTN=1            enables saving
      FOLDFLOW_ATTN_DIR=path          output directory (required)
      FOLDFLOW_ATTN_BLOCKS=all|csv    block indices to save (default: all)
      FOLDFLOW_ATTN_STEPS=csv         reverse-step indices to save
                                       (default: 50,25,10,5,2  — note step=1 is
                                       unreachable in the t>min_t loop)
      FOLDFLOW_ATTN_DESIGN_NUM=int    design index (default 0)

    `attn` is the post-softmax IPA attention with shape (B, H, L, L). It is
    permuted to (B, L, L, H) on save so the last axis indexes heads, matching
    the analyzer's contract.
    """
    if os.getenv("FOLDFLOW_SAVE_ATTN", "0") != "1":
        return
    if block_idx is None or step_idx is None:
        return
    blocks_env = os.getenv("FOLDFLOW_ATTN_BLOCKS", "all").strip()
    save_all_blocks = blocks_env.lower() == "all"
    if not save_all_blocks:
        save_blocks = {int(x) for x in blocks_env.split(",") if x.strip()}
        if int(block_idx) not in save_blocks:
            return
    save_steps = {int(x) for x in os.getenv("FOLDFLOW_ATTN_STEPS", "50,25,10,5,2").split(",")}
    s_val = int(step_idx.item()) if hasattr(step_idx, "item") else int(step_idx)
    if s_val not in save_steps:
        return
    out_dir = os.getenv("FOLDFLOW_ATTN_DIR")
    if not out_dir:
        return
    d_num = int(os.getenv("FOLDFLOW_ATTN_DESIGN_NUM", "0"))
    d_dir = os.path.join(out_dir, f"design_{d_num}")
    os.makedirs(d_dir, exist_ok=True)
    fpath = os.path.join(d_dir, f"attn_step{s_val}_block_{int(block_idx)}.pt")
    torch.save(attn.detach().permute(0, 2, 3, 1).cpu().half(), fpath)

# FIXME: matrix_to_axis_angle is only called in `forward_gt` which does not look to be used ever. Clean this once the inference and train are working.
# from torch_Riemannian_cfm.so3_utils.so3_helpers import matrix_to_axis_angle

from zeus_foldflow.data import all_atom
from zeus_foldflow.data import utils as du
from zeus_foldflow.utils.symmetry_utils import apply_symmetry_general, get_symmetry_order
from openfold.utils import rigid_utils as ru
from openfold.utils.rigid_utils import Rigid

# from layers import CrossAttentionTransformerEncoderLayer


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(
        calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1)
    )
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(
        ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1)
    )
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3), ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return phi_angles


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
        self,
        *,
        node_embed_size,
        edge_embed_in,
        edge_embed_out,
        num_layers=2,
        node_dilation=2,
    ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed, symm_n=None):
        # symm_n is unused here (only EfficientEdgeTransition needs it). It is
        # accepted so the IpaNetwork trunk can pass the same kwargs to both
        # versions uniformly.
        del symm_n
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            axis=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(
            batch_size * num_res**2, -1
        )
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed


class EfficientEdgeTransition(nn.Module):
    """Subunit-only edge transition. Verbatim port of FrameDiff
    ``EfficientEdgeTransition`` (model/ipa_pytorch.py).

    Parameter shapes are identical to ``EdgeTransition`` so existing weights
    can be loaded into either class without retraining.
    """

    def __init__(
        self,
        *,
        node_embed_size,
        edge_embed_in,
        edge_embed_out,
        num_layers=2,
        node_dilation=2,
    ):
        super(EfficientEdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed, symm_n):
        # Duplicate node_embed according to symmetry
        symm_node_embed = node_embed.repeat(1, symm_n, 1)
        node_embed = self.initial_embed(node_embed)
        symm_node_embed = self.initial_embed(symm_node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, symm_n * num_res, 1)),
                torch.tile(symm_node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            axis=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(
            batch_size * symm_n * num_res**2, -1
        )
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, symm_n * num_res, -1)
        return edge_embed


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_rbf = Linear(20, 1)
        self.linear_rbf.weight.requires_grad = False
        self.linear_rbf.bias.requires_grad = False

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        symm_n: Optional[int] = None,
        symmetry_type: str = "C",
        symmetry_origin_translation: Optional["np.ndarray"] = None,
        face_offset_radius: float = 0.0,
        coordinate_scaling: float = 1.0,
        block_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
            symm_n / symmetry_type / symmetry_origin_translation /
                face_offset_radius / coordinate_scaling:
                Accepted (and ignored) so the symmetric IpaNetwork trunk can
                pass the same kwargs to either ``InvariantPointAttention`` or
                ``EfficientInvariantPointAttention`` uniformly. The non-
                efficient path expands rigids outside the IPA layer, so these
                kwargs are no-ops here.
            block_idx / step_idx:
                Used by the env-gated _save_attn_tensor hook to dump
                post-softmax attention. Both None (default) skips saving.
        Returns:
            [*, N_res, C_s] single representation update
        """
        del symm_n, symmetry_type, symmetry_origin_translation
        del face_offset_radius, coordinate_scaling
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)
        _save_attn_tensor(a, block_idx, step_idx)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat(o_feats, dim=-1).to(dtype=z[0].dtype))

        return s


class EfficientInvariantPointAttention(nn.Module):
    """Subunit-only IPA used when ``efficient_symmetry=True``. The single
    representation ``s`` and rigids ``r`` arrive sliced to the first subunit
    only; this layer expands them via ``apply_symmetry_general`` for the K/V
    projection so attention is computed on (L_asu queries) x (L_asu*symm_n
    keys/values). The Q projection / output stays subunit-shaped.

    Parameter shapes are byte-identical to ``InvariantPointAttention`` so
    existing FoldFlow weights load into either class without retraining.
    """

    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super(EfficientInvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        # Kept for parameter-shape compatibility with the regular IPA.
        self.linear_rbf = Linear(20, 1)
        self.linear_rbf.weight.requires_grad = False
        self.linear_rbf.bias.requires_grad = False

    def forward(
        self,
        s: torch.Tensor,  # [B, L_asu, C_s]
        z: Optional[torch.Tensor],  # [B, L_asu, L_asu*symm_n, C_z]
        r: Rigid,  # [B, L_asu]
        mask: torch.Tensor,  # [B, L_asu]
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        symm_n: Optional[int] = None,
        symmetry_type: str = "C",
        symmetry_origin_translation: Optional["np.ndarray"] = None,
        face_offset_radius: float = 0.0,
        coordinate_scaling: float = 1.0,
    ) -> torch.Tensor:
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        # Duplicate node embeddings according to symmetry.
        symm_s = s.repeat(1, symm_n, 1)
        q_mask = mask
        k_mask = mask.repeat(1, symm_n)
        # ``r`` here is the scaled subunit rigid; pass ``coordinate_scaling``
        # through so ``apply_symmetry_general`` scales the pivot into the same
        # space before expanding.
        symm_r = apply_symmetry_general(
            r,
            symmetry_type,
            symm_n,
            origin_translation=symmetry_origin_translation,
            face_offset_radius=face_offset_radius,
            coordinate_scaling=coordinate_scaling,
        )

        q = self.linear_q(s)
        kv = self.linear_kv(symm_s)

        # [*, L, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, L_asu, H * P_q * 3]
        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, L_asu*symm_n, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(symm_s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = symm_r[..., None].apply(kv_pts)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        # [*, L_asu, L_asu*symm_n, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            z[0] = z[0].cpu()

        # [*, H, L_asu, L_asu*symm_n]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2
        pt_att = pt_att.sum(dim=-1)
        head_weights = self.softplus(self.head_weights).view(
            *([1] * (pt_att.ndim - 2)), -1, 1
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        attn_mask = q_mask.unsqueeze(-1) * k_mask.unsqueeze(-2)
        attn_mask = self.inf * (attn_mask - 1)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + attn_mask.unsqueeze(-3)
        a = self.softmax(a)

        # [*, L_asu, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)

        # [*, H, 3, L_asu, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        s = self.linear_out(torch.cat(o_feats, dim=-1).to(dtype=z[0].dtype))
        return s


class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_3.weight.requires_grad = False
        self.linear_3.bias.requires_grad = False

        self.linear_final = Linear(self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class ScoreLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(ScoreLayer, self).__init__()

        self.linear_1 = Linear(dim_in, dim_hid, init="relu")
        self.linear_2 = Linear(dim_hid, dim_hid)
        self.linear_3 = Linear(dim_hid, dim_out, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = s + s_initial
        s = self.linear_3(s)
        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class MLP(torch.nn.Module):
    def __init__(self, dim, time_dim=1, out_dim=None, w=512, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.te = PositionalEmbedding(time_dim, "sinusoidal")
        if out_dim is None:
            out_dim = dim
        self.combiner = torch.nn.Linear(dim + (time_dim if time_varying else 0), w)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(w, w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, t):
        pt = self.te(t).squeeze()
        x = self.combiner(torch.cat([x, pt], dim=-1))
        x = torch.nn.ReLU()(x)
        return self.net(x)
        # return self.net(x) * 10


class IpaNetwork(nn.Module):
    def __init__(self, model_conf, flow_matcher):
        super(IpaNetwork, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.flow_matcher = flow_matcher
        self.update_edge_all = getattr(ipa_conf, "do_last_edge_update", False)

        # --- Symmetry config (no-op when symm_n=1) ---
        self._symmetry_type = model_conf.get("symmetry_type", "C")
        self.symm_n = get_symmetry_order(
            self._symmetry_type, model_conf.get("symmetry", 1)
        )
        self._efficient_symmetry = model_conf.get("efficient_symmetry", False)
        # ``symmetry_origin_translation`` and ``face_offset_radius`` are
        # interpreted as user-space Angstroms. ``apply_symmetry_general``
        # takes a ``coordinate_scaling`` argument that scales the pivot
        # internally to match whatever space (scaled vs unscaled) the rigids
        # being symmetrized live in; each call site below passes the matching
        # value.
        sym_origin = model_conf.get("symmetry_origin_translation", [0.0, 0.0, 0.0])
        self._symmetry_origin_translation = np.array(list(sym_origin))
        self._face_offset_radius = float(model_conf.get("face_offset_radius", 0.0))
        self._coord_scaling = ipa_conf.coordinate_scaling

        # self.context_embed_size = ipa_conf.context_embed_size

        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        for b in range(ipa_conf.num_blocks):
            if self._efficient_symmetry:
                self.trunk[f"ipa_{b}"] = EfficientInvariantPointAttention(ipa_conf)
            else:
                self.trunk[f"ipa_{b}"] = InvariantPointAttention(ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(ipa_conf.c_s)
            self.trunk[f"skip_embed_{b}"] = Linear(
                self._model_conf.node_embed_size, self._ipa_conf.c_skip, init="final"
            )
            tfmr_in = ipa_conf.c_s + self._ipa_conf.c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )

            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, ipa_conf.seq_tfmr_num_layers
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(tfmr_in, ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(
                c=ipa_conf.c_s
            )
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(ipa_conf.c_s)

            if b < ipa_conf.num_blocks - 1 or self.update_edge_all:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                if self._efficient_symmetry:
                    self.trunk[f"edge_transition_{b}"] = EfficientEdgeTransition(
                        node_embed_size=ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )
                else:
                    self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                        node_embed_size=ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )

        self.torsion_pred = TorsionAngles(ipa_conf.c_s, 1)

    def forward(self, init_node_embed, edge_embed, input_feats, use_context=False, encoder_mode=False):
        node_mask = input_feats["res_mask"].type(torch.float32)
        flow_mask = (1 - input_feats["fixed_mask"].type(torch.float32)) * node_mask

        # Subunit size: when symm_n=1 this is the full N, so all the slicing
        # below collapses to a no-op.
        subunit_size = init_node_embed.shape[1] // self.symm_n

        init_frames = input_feats["rigids_t"].type(torch.float32)
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)

        # In the non-efficient symmetric path we re-symmetrize the rigids from
        # the first subunit before running the trunk.  This runs AFTER
        # ``scale_rigids``, so ``curr_rigids`` is in scaled space and gets
        # ``coordinate_scaling=self._coord_scaling`` for the pivot; ``init_rigids``
        # was never scaled and uses ``coordinate_scaling=1.0``.  When symm_n=1
        # the expansion is mathematically the identity, so existing FoldFlow
        # models are unaffected.
        if not self._efficient_symmetry:
            curr_rigids = apply_symmetry_general(
                curr_rigids[:, :subunit_size],
                self._symmetry_type,
                self.symm_n,
                origin_translation=self._symmetry_origin_translation,
                face_offset_radius=self._face_offset_radius,
                coordinate_scaling=self._coord_scaling,
            )
            init_rigids = apply_symmetry_general(
                init_rigids[:, :subunit_size],
                self._symmetry_type,
                self.symm_n,
                origin_translation=self._symmetry_origin_translation,
                face_offset_radius=self._face_offset_radius,
                coordinate_scaling=1.0,
            )

        if self._efficient_symmetry:  # only run compute on the first subunit
            edge_mask = node_mask[:, :subunit_size, None] * node_mask[..., None, :]
            init_node_embed = init_node_embed * node_mask[..., None]
            node_embed = init_node_embed * node_mask[..., None]
            init_node_embed = init_node_embed[:, :subunit_size, :]
            node_embed = node_embed[:, :subunit_size, :]
            node_mask = node_mask[:, :subunit_size]
            curr_rigids = curr_rigids[:, :subunit_size]
            init_rigids = init_rigids[:, :subunit_size]
            flow_mask = flow_mask[:, :subunit_size]
        else:
            edge_mask = node_mask[..., None] * node_mask[..., None, :]
            init_node_embed = init_node_embed * node_mask[..., None]
            node_embed = init_node_embed * node_mask[..., None]

        # Only the efficient IPA uses the pivot/face_radius kwargs (it calls
        # apply_symmetry_general internally on scaled subunit rigids); the
        # regular IPA does not, so we skip passing them to keep signatures
        # narrow.
        ipa_kwargs = dict(
            symm_n=self.symm_n,
            symmetry_type=self._symmetry_type,
        )
        if self._efficient_symmetry:
            ipa_kwargs["symmetry_origin_translation"] = (
                self._symmetry_origin_translation
            )
            ipa_kwargs["face_offset_radius"] = self._face_offset_radius
            ipa_kwargs["coordinate_scaling"] = self._coord_scaling

        attn_step_idx = (
            input_feats.get("attn_step_idx", None)
            if isinstance(input_feats, dict)
            else None
        )
        for b in range(self._ipa_conf.num_blocks):
            block_kwargs = dict(ipa_kwargs)
            if not self._efficient_symmetry:
                block_kwargs["block_idx"] = b
                block_kwargs["step_idx"] = attn_step_idx
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
                **block_kwargs,
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat(
                [node_embed, self.trunk[f"skip_embed_{b}"](init_node_embed)], dim=-1
            )

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * flow_mask[..., None]
            )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, flow_mask[..., None]
            )

            if b < self._ipa_conf.num_blocks - 1 or self.update_edge_all:
                edge_embed = self.trunk[f"edge_transition_{b}"](
                    node_embed, edge_embed, symm_n=self.symm_n
                )
                edge_embed *= edge_mask[..., None]

        # In efficient mode the trunk only ran on the first subunit; mirror the
        # results back out to the full symmetric set so that the rotation and
        # translation vector fields below have shape [B, symm_n*L_asu, ...].
        # ``init_rigids`` lives in user-space (coordinate_scaling=1.0);
        # ``curr_rigids`` lives in scaled space (coordinate_scaling=self._coord_scaling).
        if self._efficient_symmetry:
            init_rigids = apply_symmetry_general(
                init_rigids,
                self._symmetry_type,
                self.symm_n,
                origin_translation=self._symmetry_origin_translation,
                face_offset_radius=self._face_offset_radius,
                coordinate_scaling=1.0,
            )
            curr_rigids = apply_symmetry_general(
                curr_rigids,
                self._symmetry_type,
                self.symm_n,
                origin_translation=self._symmetry_origin_translation,
                face_offset_radius=self._face_offset_radius,
                coordinate_scaling=self._coord_scaling,
            )
            node_embed = node_embed.repeat(1, self.symm_n, 1)
            node_mask = node_mask.repeat(1, self.symm_n)
            flow_mask = flow_mask.repeat(1, self.symm_n)

        t = input_feats["t"].requires_grad_(True)

        # In symmetric (non-efficient) mode the trunk operates on the full
        # set of residues, but the per-residue ``bb_update`` in each block
        # produces asymmetric updates that drift the predicted ``curr_rigids``
        # away from a perfectly symmetric configuration.  Project both
        # ``node_embed`` and ``curr_rigids`` back onto the symmetric set so
        # that downstream consumers (psi prediction, atom37 reconstruction,
        # the post-step inference-loop slice + apply_symmetry) all see
        # symmetric inputs.  This is a no-op when ``symm_n == 1``.
        if not self._efficient_symmetry and self.symm_n > 1:
            node_embed = node_embed[:, :subunit_size].repeat(1, self.symm_n, 1)
            curr_rigids = apply_symmetry_general(
                curr_rigids[:, :subunit_size],
                self._symmetry_type,
                self.symm_n,
                origin_translation=self._symmetry_origin_translation,
                face_offset_radius=self._face_offset_radius,
                coordinate_scaling=self._coord_scaling,
            )

        # ipdb.set_trace()
        _, rot_vectorfield = self.flow_matcher.calc_rot_vectorfield(
            curr_rigids.get_rots().get_rot_mats(),
            init_rigids.get_rots().get_rot_mats(),
            t,
        )
        rot_vectorfield = rot_vectorfield * node_mask[..., None, None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            "psi": psi_pred,
            "final_rigids": curr_rigids,
        }
        if encoder_mode:
            model_out["node_embed"] = node_embed
            model_out["edge_embed"] = edge_embed
            return model_out
        trans_vectorfield = self.flow_matcher.calc_trans_vectorfield(
            curr_rigids.get_trans(),
            init_rigids.get_trans(),
            input_feats["t"][:, None, None],
            use_torch=True,
        )
        trans_vectorfield = trans_vectorfield * node_mask[..., None]
        model_out["rot_vectorfield"] = rot_vectorfield
        model_out["trans_vectorfield"] = trans_vectorfield
        return model_out
