"""Sampling routines extracted from the original Experiment; no training dependencies."""
import os
os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
import copy
import numpy as np
import torch
from openfold.utils import rigid_utils as ru
from .data import all_atom
from .data import utils as du
from .utils.symmetry_utils import apply_symmetry_general, get_symmetry_order
from .models import se3_fm
from .models.components import network

class Runner:
    def __init__(self, conf):
        self._conf = conf
        self._model_conf = conf.model
        self._data_conf = conf.data
        torch.set_default_dtype(torch.float32)
        self.flow_matcher = se3_fm.SE3FlowMatcher(conf.flow_matcher)
        self.model = network.VectorFieldNetwork(conf.model, self.flow_matcher)

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        (
            rot_vectorfield_scaling,
            trans_vectorfield_scaling,
        ) = self.flow_matcher.vectorfield_scaling(t)
        feats["rot_vectorfield_scaling"] = rot_vectorfield_scaling * t_placeholder
        feats["trans_vectorfield_scaling"] = trans_vectorfield_scaling * t_placeholder
        return feats

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
        context=None,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]
        # dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        # ----- Symmetry config (no-op when symm_n=1) -----
        symmetry_type = self._model_conf.get("symmetry_type", "C")
        symm_n = get_symmetry_order(
            symmetry_type, self._model_conf.get("symmetry", 1)
        )
        if symmetry_type in ("T", "O", "I"):
            assert self._model_conf.get("diffuse_subunit_only", False), (
                f"Polyhedral symmetry '{symmetry_type}' requires diffuse_subunit_only=true"
            )
        diffuse_subunit_only = self._model_conf.get("diffuse_subunit_only", False)
        sym_origin = self._model_conf.get(
            "symmetry_origin_translation", [0.0, 0.0, 0.0]
        )
        # ``symmetry_origin_translation`` and ``face_offset_radius`` are in
        # user-space Angstroms. The reverse-step rigids that go into
        # ``apply_symmetry_general`` below are also in user-space, so we pass
        # the pivot directly (``coordinate_scaling=1.0`` is the default).
        symm_origin_translation = np.array(list(sym_origin))
        face_offset_radius = float(self._model_conf.get("face_offset_radius", 0.0))

        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                # Skip attn dump for the self-conditioning warm pass.
                sample_feats["attn_step_idx"] = None
                sample_feats = self._self_conditioning(sample_feats)
            for step_i, t in enumerate(reverse_steps):
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    # step_idx counts down from num_t (high noise) to 1 (near
                    # clean), so the analyzer's plot titles read "step 50" at
                    # t≈1 and "step 2" at t≈min_t (the t>min_t guard skips
                    # step=1 — same convention used in FrameDiff).
                    sample_feats["attn_step_idx"] = num_t - step_i
                    model_out = self.model(sample_feats)
                    rot_vectorfield = model_out["rot_vectorfield"]
                    trans_vectorfield = model_out["trans_vectorfield"]
                    rigid_pred = model_out["rigids"]
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                    flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

                    # Symmetric reverse step: integrate only the first subunit,
                    # then re-tile the update via apply_symmetry_general so the
                    # full set of rigids stays exactly symmetric.
                    N = sample_feats["rigids_t"].shape[1]
                    assert N % symm_n == 0, (
                        f"sample_length={N} not divisible by symmetry order {symm_n}"
                    )
                    subunit_n = N // symm_n
                    diff_n = subunit_n if diffuse_subunit_only else N

                    rots_t, trans_t, rigids_t_subunit = self.flow_matcher.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(
                            sample_feats["rigids_t"][:, :diff_n, :]
                        ),
                        rot_vectorfield=du.move_to_np(
                            rot_vectorfield[:, :diff_n]
                        ),
                        trans_vectorfield=du.move_to_np(
                            trans_vectorfield[:, :diff_n]
                        ),
                        flow_mask=du.move_to_np(flow_mask[:, :diff_n]),
                        t=t,
                        dt=dt,
                        center=True,
                        noise_scale=noise_scale,
                    )
                    if not diffuse_subunit_only:
                        rigids_t_subunit = rigids_t_subunit[:, :subunit_n]
                    rigids_t = apply_symmetry_general(
                        rigids_t_subunit,
                        symmetry_type,
                        symm_n,
                        origin_translation=symm_origin_translation,
                        face_offset_radius=face_offset_radius,
                    )
                else:
                    # Final clean step (t == min_t): take the model's predicted
                    # rigids directly. FrameDiff uses ``rigid_pred`` as-is here,
                    # which assumes the model produces symmetric output. For
                    # FoldFlow checkpoints not retrained for symmetric inputs,
                    # the trunk's per-residue ``bb_update`` causes the output
                    # rigids to drift away from symmetry by ~0.1A.  We project
                    # back onto the symmetric set by slicing the first subunit
                    # and re-applying ``apply_symmetry_general`` (a no-op when
                    # symm_n=1).
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)
                    rot_vectorfield = model_out["rot_vectorfield"]
                    trans_vectorfield = model_out["trans_vectorfield"]
                    rigid_pred = model_out["rigids"]
                    fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                    flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]
                    N = rigid_pred.shape[1]
                    subunit_n = N // symm_n
                    rigids_t = apply_symmetry_general(
                        ru.Rigid.from_tensor_7(rigid_pred[:, :subunit_n]),
                        symmetry_type,
                        symm_n,
                        origin_translation=symm_origin_translation,
                        face_offset_radius=face_offset_radius,
                    )

                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from vectorfield predictions.
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )
                psi_pred = model_out["psi"]
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        return ret
