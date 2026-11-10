"""Sampling routines extracted from the original Experiment; no training dependencies."""
import os
os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
import copy
import numpy as np
import torch
from openfold.utils import rigid_utils as ru
from .data import all_atom
from .data import utils as du
from .data.symmetry_utils import apply_symmetry_general, get_symmetry_order
from .data import se3_diffuser
from .model import score_network

class Runner:
    def __init__(self, conf):
        self._conf = conf
        self._model_conf = conf.model
        self._data_conf = conf.data
        self.diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
        self.model = score_network.ScoreNetwork(conf.model, self.diffuser)

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch['sc_ca_t'] = model_sc['rigids'][..., 4:]
        return batch

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=False,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder)
                # Skip attn dump for the self-conditioning warm pass.
                sample_feats['attn_step_idx'] = None
                sample_feats = self._self_conditioning(sample_feats)
            for step_i, t in enumerate(reverse_steps):
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    # step_idx counts down from num_t (high noise) to 1 (near-clean),
                    # mirroring RFdiffusion's timestep convention so the analyzer
                    # plot titles read "step 500" at t≈1 and "step 1" at t≈min_t.
                    sample_feats['attn_step_idx'] = num_t - step_i
                    model_out = self.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    if self._model_conf.embed.embed_self_conditioning:
                        sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    # rigids_t = self.diffuser.reverse(
                    #     rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                    #     rot_score=du.move_to_np(rot_score),
                    #     trans_score=du.move_to_np(trans_score),
                    #     diffuse_mask=du.move_to_np(diffuse_mask),
                    #     t=t,
                    #     dt=dt,
                    #     center=center,
                    #     noise_scale=noise_scale,
                    # )
                    symmetry_type = self._model_conf.get("symmetry_type", "C")
                    if symmetry_type in ("T", "O", "I"):
                        assert self._model_conf.diffuse_subunit_only, \
                            f"Polyhedral symmetry '{symmetry_type}' requires diffuse_subunit_only=true"
                    if symmetry_type == "H":
                        assert self._model_conf.diffuse_subunit_only and self._model_conf.efficient_symmetry, \
                            "Helical symmetry requires diffuse_subunit_only=true and efficient_symmetry=true (open orbit)"
                    if symmetry_type == "L":
                        assert self._model_conf.diffuse_subunit_only and self._model_conf.efficient_symmetry, \
                            "Lattice symmetry requires diffuse_subunit_only=true and efficient_symmetry=true (open orbit)"
                    if symmetry_type == "L3":
                        assert self._model_conf.diffuse_subunit_only and self._model_conf.efficient_symmetry, \
                            "3D lattice symmetry requires diffuse_subunit_only=true and efficient_symmetry=true (open orbit)"
                    symm_n = get_symmetry_order(
                        symmetry_type, self._conf.model.symmetry,
                        self._model_conf.get("inner_symmetry_type", None),
                        self._model_conf.get("inner_symmetry", None),
                    )
                    subunit_n = sample_feats['rigids_t'].shape[1] // symm_n
                    origin_translation = np.array(self._model_conf.symmetry_origin_translation)
                    if self._model_conf.diffuse_subunit_only:
                        diff_n = subunit_n
                    else:
                        diff_n = sample_feats['rigids_t'].shape[1]
                    rigids_t_subunit = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t'][:, :diff_n, :]),
                        rot_score=du.move_to_np(rot_score[:,:diff_n,:]),
                        trans_score=du.move_to_np(trans_score[:,:diff_n,:]),
                        diffuse_mask=du.move_to_np(diffuse_mask[:, :diff_n]),
                        t=t,
                        dt=dt,
                        center=True,
                        noise_scale=noise_scale,
                    )
                    # Always re-expand from subunit to enforce symmetry
                    if not self._model_conf.diffuse_subunit_only:
                        rigids_t_subunit = rigids_t_subunit[:, :subunit_n]
                    rigids_t = apply_symmetry_general(rigids_t_subunit, symmetry_type, symm_n, origin_translation=origin_translation, face_offset_radius=self._model_conf.get("face_offset_radius", 0.0), ring_offset=self._model_conf.get("ring_offset", 0.0), twist=self._model_conf.get("twist", 0.0), rise=self._model_conf.get("rise", 0.0), n_start=self._model_conf.get("n_start", 1), lattice_a=self._model_conf.get("lattice_a", [0.0, 0.0, 0.0]), lattice_b=self._model_conf.get("lattice_b", [0.0, 0.0, 0.0]), lattice_c=self._model_conf.get("lattice_c", [0.0, 0.0, 0.0]), n_a=self._model_conf.get("n_a", 1), n_b=self._model_conf.get("n_b", 1), n_c=self._model_conf.get("n_c", 1), point_group_order=self._model_conf.get("point_group_order", 1), inner_symmetry_type=self._model_conf.get("inner_symmetry_type", None), inner_symm_n=self._model_conf.get("inner_symmetry", None), inner_origin_translation=self._model_conf.get("inner_origin_translation", [0.0, 0.0, 0.0]))
                else:
                    model_out = self.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from score predictions.
                #rigids_t, psi_pred, rigid_pred, diffuse_mask, fixed_mask, gt_trans_0, sample_feats['rigids_t'] all need to be copied
                
                psi_pred = model_out['psi']
                gt_rigid = ru.Rigid.from_tensor_7(sample_feats['rigids_t']).to_tensor_7()
                
                gt_trans_0 = gt_rigid[..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred),
                        psi_pred,
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(
                    rigids_t, psi_pred)[0]
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
            'prot_traj': all_bb_prots,
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['psi_pred'] = psi_pred[None]
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret
