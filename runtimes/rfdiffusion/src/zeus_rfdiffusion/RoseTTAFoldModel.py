import torch
import torch.nn as nn
from zeus_rfdiffusion.Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from zeus_rfdiffusion.Track_module import IterativeSimulator
from zeus_rfdiffusion.AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, ExpResolvedNetwork, LDDTNetwork
from opt_einsum import contract as einsum

# Every value model.symmetry_mode may take. Anything else is rejected at
# construction rather than silently falling through to null mode -- an
# unrecognised name used to run as plain null with no warning.
VALID_SYMMETRY_MODES = (
    None,                       # stock RFdiffusion, full (L, L) pair
    'restricted',               # (L_asu, L) slab throughout
    'hybrid_restr_str',         # null MSA/pair path, restricted str2str
    'hybrid_restr_pair',        # restricted MSA/pair path, null str2str
    'null_chainbreak_ablate',   # null everywhere, but seqsep adopts the
                                # restricted slab's forward-only chain breaks
    'null_chainbreak_torsion_ablate',
                                # as null_chainbreak_ablate, and additionally
                                # removes BOTH carriers of chain-asymmetry in the
                                # featurization: tiles chain 0's alpha_t, and
                                # zeroes the N/C-terminus one-hots in the MSA.
                                # _preprocess indexes both by absolute position
                                # in the concatenated assembly, so only chain 0
                                # gets an N-terminus / defined phi and only chain
                                # n-1 a C-terminus / defined omega. That is what
                                # the restricted ASU truncation discards. Both
                                # must go: each alone buys < 5x, together ~1000x.
    'restricted_torsion_ablate',
                                # restricted architecture, with the same two
                                # featurization ablations as
                                # null_chainbreak_torsion_ablate. Pairs with it:
                                # both arms then share a chain-symmetric input,
                                # which is what an equivalence comparison needs.
)

# Modes that differ from a base mode only in featurization, not architecture.
# Submodules are constructed with the base name so existing dispatch keeps working.
ARCH_MODE = {
    'restricted_torsion_ablate': 'restricted',
}

# Modes whose input features are made chain-symmetric before the model runs.
FEATURE_SYMMETRISE_MODES = (
    'null_chainbreak_torsion_ablate',
    'restricted_torsion_ablate',
)

# Modes that carry the restricted (L_asu, L) pair slab.
RESTRICTED_PAIR_MODES = (
    'restricted', 'restricted_torsion_ablate', 'hybrid_restr_pair',
)


class RoseTTAFoldModule(nn.Module):
    def __init__(self, 
                 n_extra_block, 
                 n_main_block, 
                 n_ref_block,
                 d_msa,
                 d_msa_full,
                 d_pair,
                 d_templ,
                 n_head_msa,
                 n_head_pair,
                 n_head_templ,
                 d_hidden,
                 d_hidden_templ,
                 p_drop,
                 d_t1d,
                 d_t2d,
                 T, # total timesteps (used in timestep emb
                 use_motif_timestep, # Whether to have a distinct emb for motif
                 freeze_track_motif, # Whether to freeze updates to motif in track
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 input_seq_onehot=False,     # For continuous vs. discrete sequence
                 symmetry_mode=None,         # None for default, 'restricted' for restricted symmetric modules
                 ):

        super(RoseTTAFoldModule, self).__init__()

        if symmetry_mode == 'null':      # hydra can hand this through as a string
            symmetry_mode = None
        if symmetry_mode not in VALID_SYMMETRY_MODES:
            raise ValueError(
                f"Unknown model.symmetry_mode {symmetry_mode!r}. Valid values: "
                + ", ".join(repr(m) for m in VALID_SYMMETRY_MODES)
                + ". (An unrecognised value would otherwise run as plain null "
                  "mode with no warning.)")

        self.freeze_track_motif = freeze_track_motif
        self.symmetry_mode = symmetry_mode

        # The *_torsion_ablate modes differ from their base mode only in how the
        # input features are built, never in which modules run. Submodules are
        # therefore given the base architecture name, so every existing
        # `symmetry_mode == 'restricted'` branch in Track_module / Embeddings
        # keeps working without having to enumerate the new name in each one.
        arch_mode = ARCH_MODE.get(symmetry_mode, symmetry_mode)

        # Input Embeddings
        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d,
                                   symmetry_mode=arch_mode)


        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop,
                                            symmetry_mode=arch_mode)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa)
        self.lddt_pred = LDDTNetwork(d_state)
       
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)

    def forward(self, msa_latent, msa_full, seq, xyz, idx, t,
                t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                msa_prev=None, pair_prev=None, state_prev=None,
                return_raw=False, return_full=False, return_infer=False,
                use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None,
                cyclic_reses=None, symm_n=1, perm_table=None, sym_rots=None):

        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        emb_symm_n = symm_n if self.symmetry_mode in RESTRICTED_PAIR_MODES else 1
        # Ablation only: full (L, L) pair, but inter-chain seqsep forced forward-only.
        ablate_n = symm_n if self.symmetry_mode in (
            'null_chainbreak_ablate', 'null_chainbreak_torsion_ablate') else 1
        # Ablation only: make alpha_t chain-symmetric by tiling chain 0's rows.
        # get_torsions has no chain awareness -- it slides omega over [:, :-1]
        # and phi over [:, 1:] of the concatenated assembly -- so the global
        # first/last residue get sentinel values that no other chain has.
        if self.symmetry_mode in FEATURE_SYMMETRISE_MODES and symm_n > 1:
            L_asu = L // symm_n
            tile_idx = torch.arange(L_asu, device=alpha_t.device).repeat(symm_n)
            alpha_t = alpha_t.index_select(2, tile_idx)     # (B, T, L, 30)
            # The MSA carries a second, independent chain-asymmetry: _preprocess
            # sets the terminus one-hots by absolute index
            #   msa_masked[:, :, 0, 46] / [:, :, -1, 47]
            #   msa_full  [:, :, 0, 23] / [:, :, -1, 24]
            # so only chain 0 gets an N-terminus and only chain n-1 a C-terminus.
            # Removing only one of the two carriers leaves the ASU truncation
            # lossy -- measured, each alone buys < 5x while both together buy
            # ~1000x -- so zero these as well. Cloned: these tensors belong to
            # the caller.
            if msa_latent.shape[-1] > 47:
                msa_latent = msa_latent.clone()
                msa_latent[..., 46] = 0.0
                msa_latent[..., 47] = 0.0
            if msa_full.shape[-1] > 24:
                msa_full = msa_full.clone()
                msa_full[..., 23] = 0.0
                msa_full[..., 24] = 0.0
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, cyclic_reses,
                                                  symm_n=emb_symm_n,
                                                  chainbreak_ablate_n=ablate_n)
        msa_full = self.full_emb(msa_full, seq, idx)

        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev) #okay for restricted pairs
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle
        # Get timestep embedding (if using)
        if hasattr(self, 'timestep_embedder'):
            assert t is not None
            time_emb = self.timestep_embedder(L,t,motif_mask)
            n_tmpl = t1d.shape[1]
            t1d = torch.cat([t1d, time_emb[None,None,...].repeat(1,n_tmpl,1,1)], dim=-1)
        # add template embedding
        B, LR, LC = pair.shape[:3]
        pair, state = self.templ_emb(t1d, t2d[:, :, :LR, :LC, ...], alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint, symm_n=symm_n, perm_table=perm_table)
        # Predict coordinates from given inputs
        is_frozen_residue = motif_mask if self.freeze_track_motif else torch.zeros_like(motif_mask).bool()
        msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, msa_full, pair, xyz[:,:,:3],
                                                         state, idx, use_checkpoint=use_checkpoint,
                                                         motif_mask=is_frozen_residue, cyclic_reses=cyclic_reses, t=t,
                                                         symm_n=symm_n, perm_table=perm_table, sym_rots=sym_rots)
        #Look at residue outputs
        # torch.save(xyz, f'artifacts/xyz/xyz_time={t.item()}.pt')
        # torch.save(msa_latent, f'artifacts/msa_latent/msa_latent_time={t.item()}.pt')
        # torch.save(msa_full, f'artifacts/msa_full/msa_full_time={t.item()}.pt')
        # torch.save(msa, f'artifacts/msa/msa_time={t.item()}.pt')
        # torch.save(pair, f'artifacts/pair/pair_time={t.item()}.pt')
        # torch.save(state, f'artifacts/state/state_time={t.item()}.pt')
        if return_raw:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz, state, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)
        
        # Predict LDDT
        lddt = self.lddt_pred(state)

        if return_infer:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            
            # get scalar plddt
            nbin = lddt.shape[1]
            bin_step = 1.0 / nbin
            lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
            pred_lddt = nn.Softmax(dim=1)(lddt)
            pred_lddt = torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

            return msa[:,0], pair, xyz, state, alpha_s[-1], logits_aa.permute(0,2,1), pred_lddt

        #
        # predict distogram & orientograms
        logits = self.c6d_pred(pair)
        
        # predict experimentally resolved or not
        logits_exp = self.exp_pred(msa[:,0], state)
        
        # get all intermediate bb structures
        xyz = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)

        return logits, logits_aa, logits_exp, xyz, alpha_s, lddt
