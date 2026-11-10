"""Chain-permutation-invariant featurization for symmetric assemblies.

`Sampler._preprocess` builds two per-residue features by *absolute* index in the
concatenated assembly, which makes identical chains carry different features:

  1. The N/C-terminus one-hots are written at global residue 0 and L-1 only, so
     only chain 0 has an N-terminus and only chain n-1 a C-terminus.

  2. `util.get_torsions` slides omega over `[:-1]` and phi over `[1:]` of the
     whole assembly and masks them at the row's first/last residue. Over a
     concatenation that means chain 0 alone gets the phi sentinel, chain n-1
     alone the omega sentinel, and each interior chain junction gets a dihedral
     measured across a peptide bond that does not exist.

Restricted ("efficient") mode truncates the MSA to the ASU and tiles chain 0's
keys, values and outputs across the assembly -- it *assumes* chain k's features
equal chain 0's rather than computing them. These two carriers are exactly where
that assumption fails, which is why the two modes disagree on a structurally
symmetric input. Rebuilding both features per chain removes the discrepancy at
its source, so restricted and null see the same input and compute the same
function.

The rebuild is also the physically correct featurization: a C_n homo-oligomer is
n separate polypeptides, so every chain genuinely has its own N- and C-terminus
and no peptide bond crosses a chain boundary.
"""
import torch

from zeus_rfdiffusion import util

# Channel positions of the terminus one-hots, as written by Sampler._preprocess.
MSA_MASKED_N_TERM, MSA_MASKED_C_TERM = 46, 47
MSA_FULL_N_TERM, MSA_FULL_C_TERM = 23, 24


def chain_termini(L, n_chains):
    """Absolute indices of each chain's first and last residue.

    Returns (starts, ends), each a LongTensor of length `n_chains`. With
    n_chains=1 this is `[0]` and `[L-1]`, i.e. the stock global indices.
    """
    if L % n_chains != 0:
        raise ValueError(
            f"assembly length {L} is not divisible by {n_chains} chains")
    L_asu = L // n_chains
    starts = torch.arange(n_chains, dtype=torch.long) * L_asu
    return starts, starts + L_asu - 1


def set_chain_terminus_onehots(msa_masked, msa_full, n_chains):
    """Move the N/C-terminus one-hots onto every chain's own ends.

    Returns new tensors; the caller's are not modified. `n_chains <= 1` is a
    no-op, so a monomer keeps stock behaviour bit-for-bit.

    Correcting the flags is deliberate rather than clearing them. Zeroing all
    four channels is also chain-symmetric, but it tells the model that no
    residue anywhere is a terminus -- discarding true information to buy the
    symmetry. Every chain really does have both termini.
    """
    if n_chains <= 1:
        return msa_masked, msa_full

    L = msa_masked.shape[-2]
    starts, ends = chain_termini(L, n_chains)
    starts = starts.to(msa_masked.device)
    ends = ends.to(msa_masked.device)

    msa_masked = msa_masked.clone()
    msa_full = msa_full.clone()
    msa_masked[..., MSA_MASKED_N_TERM] = 0.0
    msa_masked[..., MSA_MASKED_C_TERM] = 0.0
    msa_full[..., MSA_FULL_N_TERM] = 0.0
    msa_full[..., MSA_FULL_C_TERM] = 0.0
    msa_masked[:, :, starts, MSA_MASKED_N_TERM] = 1.0
    msa_masked[:, :, ends, MSA_MASKED_C_TERM] = 1.0
    msa_full[:, :, starts, MSA_FULL_N_TERM] = 1.0
    msa_full[:, :, ends, MSA_FULL_C_TERM] = 1.0
    return msa_masked, msa_full


def get_torsions_per_chain(xyz, seq, n_chains,
                           torsion_indices, torsion_can_flip, ref_angles):
    """`util.get_torsions` applied to each chain independently.

    `xyz` is (B, L, 27, 3) and `seq` is (B, L); the output is (B, L, 10, 3),
    laid out as [sin, cos, mask] per torsion -- the same tensor
    `Sampler._preprocess` flattens into the last axis of `alpha_t`.

    Folding the chain axis into the batch axis is what does the work:
    `get_tor_mask` masks phi at each row's first residue and omega at its last,
    so one row per chain gives every chain its own terminus sentinels and never
    measures a dihedral across a chain junction. `n_chains=1` is exactly the
    stock single call.
    """
    B, L = xyz.shape[0], xyz.shape[1]
    if L % n_chains != 0:
        raise ValueError(
            f"assembly length {L} is not divisible by {n_chains} chains")
    L_asu = L // n_chains

    xyz_c = xyz.reshape(B * n_chains, L_asu, *xyz.shape[2:])
    seq_c = seq.reshape(B * n_chains, L_asu)

    alpha, _, alpha_mask, _ = util.get_torsions(
        xyz_c, seq_c, torsion_indices, torsion_can_flip, ref_angles)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))
    alpha[torch.isnan(alpha)] = 0.0

    out = torch.cat(
        (alpha.reshape(B * n_chains, L_asu, 10, 2),
         alpha_mask.reshape(B * n_chains, L_asu, 10, 1)), dim=-1)
    return out.reshape(B, L, 10, 3)
