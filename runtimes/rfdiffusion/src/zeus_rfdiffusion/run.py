#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle
import torch
from omegaconf import OmegaConf
import hydra
import logging
from zeus_rfdiffusion.util import writepdb_multi, writepdb, writemmcif, writemmcif_multi
from zeus_rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
import string


def pdb_chain_ids(labels):
    """Map native labels (A, ..., Z, AA, ...) to single-character PDB IDs."""
    if labels is None:
        return None
    unique = list(dict.fromkeys(labels))
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if len(unique) > len(alphabet):
        raise ValueError("PDB output supports at most 62 chains; use output_format=cif")
    mapping = dict(zip(unique, alphabet))
    return [mapping[label] for label in labels]


def make_deterministic(seed=0, cudnn=True, disable_tf32=False):
    """Pin every RNG stream this process draws from, plus the cuDNN algo choice.

    torch.manual_seed covers the CUDA generators too. The cuDNN autotuner is
    the one piece that is not seed-controlled: with benchmark on it picks an
    algorithm by timing it, so an unrelated process on the same GPU can change
    which kernel gets chosen and therefore the low-order bits of the result.

    TF32 is left alone by default. It is reproducible on a fixed GPU and much
    faster; it only costs you agreement between an Ampere-or-newer card and an
    older one, so disable_tf32 is opt-in.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import dgl
        # CUDA-enabled DGL 0.6 can abort inside CURAND when no GPU is visible.
        # CPU graph operations in this inference path do not draw DGL RNG.
        if torch.cuda.is_available():
            dgl.seed(seed)
    except Exception:
        pass  # older DGL builds have no seed(); the SE3 forward draws no RNG anyway
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _seed_for_design(conf, i_des):
    """Seed for design i. inference.seed=null keeps the historical seed = index."""
    base = getattr(conf.inference, "seed", None)
    return int(i_des) if base is None else int(base) + int(i_des)


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    overall_start_time = time.perf_counter()
    if conf.inference.deterministic:
        make_deterministic(
            _seed_for_design(conf, 0),
            cudnn=getattr(conf.inference, "deterministic_cudnn", True),
            disable_tf32=getattr(conf.inference, "disable_tf32", False),
        )

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    # Pick writer + extension once up front; default to PDB if unset.
    output_format = str(getattr(conf.inference, "output_format", "pdb")).lower()
    if output_format not in ("pdb", "cif"):
        raise ValueError(
            f"inference.output_format must be 'pdb' or 'cif', got {output_format!r}"
        )
    ext = output_format
    single_writer = writemmcif if output_format == "cif" else writepdb
    multi_writer = writemmcif_multi if output_format == "cif" else writepdb_multi

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + f"*.{ext}")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(rf".*_(\d+)\.{ext}$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(
                _seed_for_design(conf, i_des),
                cudnn=getattr(conf.inference, "deterministic_cudnn", True),
                disable_tf32=getattr(conf.inference, "disable_tf32", False),
            )
        # Separate stream, separate rewind: the NoiseProvider's generators are
        # private and the reseeding above cannot reach them.
        if hasattr(sampler, "reseed_noise"):
            sampler.reseed_noise(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(f"{out_prefix}.{ext}"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.{ext} already exists."
            )
            continue

        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        step_times_s = []
        diffusion_start_time = time.perf_counter()
        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start_time = time.perf_counter()
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_elapsed_s = time.perf_counter() - step_start_time
            step_times_s.append((int(t), step_elapsed_s))
            log.info(f"Diffusion step t={t} took {step_elapsed_s:.3f} s")
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension
        diffusion_elapsed_s = time.perf_counter() - diffusion_start_time
        if step_times_s:
            only_times = [x[1] for x in step_times_s]
            log.info(
                "Diffusion timing summary: total %.3f s over %d steps (mean %.3f s, min %.3f s, max %.3f s)",
                diffusion_elapsed_s,
                len(step_times_s),
                float(np.mean(only_times)),
                float(np.min(only_times)),
                float(np.max(only_times)),
            )

        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(
            denoised_xyz_stack,
            [
                0,
            ],
        )
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(
            px0_xyz_stack,
            [
                0,
            ],
        )

        # For logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)

        # Save outputs
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        # Output glycines, except for motif region
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # 7 is glycine

        bfacts = torch.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        # pX0 last step
        out = f"{out_prefix}.{ext}"
        output_chain_ids = pdb_chain_ids(sampler.chain_idx) if output_format == "pdb" else sampler.chain_idx

        # Now don't output sidechains
        single_writer(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=output_chain_ids,
            bfacts=bfacts,
            idx_pdb=sampler.idx_pdb
        )

        # run metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            output_chain_ids=output_chain_ids,
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
            diffusion_time_s=diffusion_elapsed_s,
            diffusion_step_times_s=step_times_s,
            diffusion_step_time_mean_s=float(np.mean([x[1] for x in step_times_s])) if step_times_s else None,
            diffusion_step_time_min_s=float(np.min([x[1] for x in step_times_s])) if step_times_s else None,
            diffusion_step_time_max_s=float(np.max([x[1] for x in step_times_s])) if step_times_s else None,
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        # Persist trajectory log (drift experiment) and noise recording (if any).
        if hasattr(sampler, "dump_trajectory_log"):
            sampler.dump_trajectory_log()
        _np = getattr(sampler, "noise_provider", None)
        if _np is not None and getattr(sampler, "_noise_mode", "off") == "record":
            _np.dump(sampler._noise_path)
            log.info(f"Noise recording saved to {sampler._noise_path}")

        if sampler.inf_conf.write_trajectory:
            # trajectory pdbs
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

            out = f"{traj_prefix}_Xt-1_traj.{ext}"
            multi_writer(
                out,
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=output_chain_ids,
            )

            out = f"{traj_prefix}_pX0_traj.{ext}"
            multi_writer(
                out,
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=output_chain_ids,
            )

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")
    log.info(f"Finished all designs in {(time.perf_counter()-overall_start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
