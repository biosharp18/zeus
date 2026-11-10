import math
from openfold.utils.rigid_utils import Rigid, Rotation
import torch
import numpy as np


def build_group_permutations(sym_rots):
    """Build the g⁻¹h permutation table for a finite rotation group.

    Given S rotation matrices [R_0, R_1, ..., R_{S-1}] (with R_0 = identity),
    returns an (S, S) integer tensor where entry [g, h] is the index k such
    that R_k = R_g⁻¹ @ R_h. Identity must be at index 0; ordering must match
    the chain layout produced by ``apply_symmetry`` / ``_apply_group_symmetry``
    (chain g lives at slice [g*L_asu : (g+1)*L_asu] in the assembled L tensor).

    Returns:
        perm_table: (S, S) LongTensor — perm_table[g, h] = index of R_g⁻¹ R_h.
        margins:    (S, S) float tensor — Frobenius gap between best/2nd-best.
    """
    rots = np.stack([
        r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        for r in sym_rots
    ]).astype(np.float64)  # (S, 3, 3)
    S = len(rots)

    perm_table = np.zeros((S, S), dtype=np.int64)
    margins = np.zeros((S, S), dtype=np.float64)
    for g in range(S):
        Rg_inv = rots[g].T  # rotation inverse = transpose
        for h in range(S):
            product = Rg_inv @ rots[h]
            diffs = np.linalg.norm(rots - product[None], axis=(1, 2))  # (S,)
            sorted_idx = np.argsort(diffs)
            perm_table[g, h] = sorted_idx[0]
            margins[g, h] = diffs[sorted_idx[1]] - diffs[sorted_idx[0]]
    return torch.from_numpy(perm_table).long(), torch.from_numpy(margins).float()


def symmetric_block_tile(x, perm_table):
    """Tile a block-row (B, LR, LC, D...) into a full pair matrix (B, LC, LC, D...).

    For cyclic groups the permutation table is circulant and this is equivalent
    to a roll. For polyhedral groups each row is a different permutation.
    """
    S = perm_table.shape[0]
    LR = x.shape[1]
    rows = []
    for g in range(S):
        perm = perm_table[g]  # (S,)
        col_idx = (perm[:, None] * LR + torch.arange(LR, device=x.device)[None, :]).reshape(-1)
        rows.append(x[:, :, col_idx])
    return torch.cat(rows, dim=1)


def get_symmetry_rotation_list(symmetry_type, symm_n, origin_translation,
                               face_offset_radius, device, dtype):
    """Return the ordered rotation list [R_0=I, ..., R_{S-1}] used to expand
    chains in null-mode. Ordering matches ``apply_symmetry`` (cyclic) and
    ``_apply_group_symmetry`` (T/O/I) so that chain g in L=G*L_asu corresponds
    to rotation R_g.

    Used to build the perm_table consumed by ``symmetric_block_tile``.
    Coordinate scaling is fixed at 1.0 because the perm_table only depends on
    rotations (offsets are irrelevant).
    """
    if symmetry_type == "C":
        rots = []
        for i in range(symm_n):
            angle = 2 * math.pi * i / symm_n
            R = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )
            rots.append(R)
        return torch.stack(rots)
    if symmetry_type == "T":
        R, _ = _tetrahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, 1.0)
        return R
    if symmetry_type == "O":
        R, _ = _octahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, 1.0)
        return R
    if symmetry_type == "I":
        R, _ = _icosahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, 1.0)
        return R
    raise NotImplementedError(
        f"Rotation list not available for symmetry_type='{symmetry_type}'.")


def _resolve_origin_translation(
    curr_rigids: Rigid,
    origin_translation: torch.Tensor = None,
    offset_radius: float = None,
) -> torch.Tensor:
    """Resolve translation pivot to shape [1, 1, 3] on the correct device/dtype."""
    if origin_translation is None and offset_radius is not None:
        origin_translation = torch.tensor([offset_radius, 0.0, 0.0]).unsqueeze(0).unsqueeze(0)
    if origin_translation is None:
        return None
    return origin_translation.to(device=curr_rigids._rots.device, dtype=curr_rigids._trans.dtype)


def apply_symmetry(
    curr_rigids: Rigid,
    symm_n: int,
    origin_translation: np.ndarray = np.array([0, 0, 0]),
    coordinate_scaling: float = 1.0,
):  # 4.8 for c30 symmetry
    """Apply cyclic C_n symmetry to a batch of rigids.

    ``coordinate_scaling`` scales the pivot to match the rigid's unit convention.
    Use 1.0 when ``curr_rigids`` is unscaled (Angstroms); use
    ``ipa_conf.coordinate_scaling`` (e.g. 0.1) when ``curr_rigids`` has been
    passed through ``scale_rigids``.
    """
    origin_translation = torch.tensor(origin_translation) * coordinate_scaling
    origin_translation = origin_translation.unsqueeze(0).unsqueeze(0)
    origin_translation = _resolve_origin_translation(curr_rigids, origin_translation=origin_translation)

    symm_translations = []
    symm_rotations = []
    for i in range(symm_n):
        angle = 2 * math.pi * i / symm_n
        rot_mat = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=curr_rigids._rots.device,
            dtype=curr_rigids._trans.dtype,
        )

        if origin_translation is not None:
            # Translate first, rotate, then translate back.
            new_trans = (
                torch.einsum("ki,bni->bnk", rot_mat, curr_rigids._trans - origin_translation)
                + origin_translation
            )
        else:
            new_trans = torch.einsum("ki,bni->bnk", rot_mat, curr_rigids._trans)

        new_rots = torch.einsum("ki,bnil->bnkl", rot_mat, curr_rigids._rots.get_rot_mats())
        symm_translations.append(new_trans)
        symm_rotations.append(new_rots)

    return Rigid(
        Rotation(rot_mats=torch.cat(symm_rotations, dim=1)),
        torch.cat(symm_translations, dim=1),
    )


_GROUP_ORDERS = {"T": 12, "O": 24, "I": 60}


def get_symmetry_order(symmetry_type, symm_n=None):
    """Return the number of group elements for the given symmetry type."""
    if symmetry_type == "C":
        return symm_n
    return _GROUP_ORDERS[symmetry_type]


# ---------------------------------------------------------------------------
# Tetrahedral symmetry helpers
# ---------------------------------------------------------------------------

def _rodrigues_matrix(k, angle):
    """Rotation matrix from unit axis k and angle via Rodrigues' formula.

    Args:
        k: [3] unit vector (torch tensor).
        angle: rotation angle in radians (float).
    Returns:
        [3, 3] rotation matrix on the same device/dtype as k.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    K = torch.zeros(3, 3, device=k.device, dtype=k.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    I3 = torch.eye(3, device=k.device, dtype=k.dtype)
    return c * I3 + (1 - c) * torch.outer(k, k) + s * K


def _choose_perpendicular(v):
    """Return a unit vector perpendicular to unit vector v.

    Deterministic: picks the coordinate axis most perpendicular to v.
    """
    abs_v = torch.abs(v)
    if abs_v[0] <= abs_v[1] and abs_v[0] <= abs_v[2]:
        e = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype)
    elif abs_v[1] <= abs_v[2]:
        e = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype)
    else:
        e = torch.tensor([0.0, 0.0, 1.0], device=v.device, dtype=v.dtype)
    perp = torch.linalg.cross(v, e)
    return perp / torch.linalg.norm(perp)


def _tetrahedral_transforms(origin_translation, face_offset_radius, device, dtype,
                            coordinate_scaling: float = 1.0):
    """Compute rotation matrices and offsets for tetrahedral symmetry.

    Two-step construction:
      1. Face 0 (base): C3 around origin_translation axis, pivot at
         face_offset_radius perpendicular to that axis. Identity stays at origin.
      2. Faces 1-3: rotate entire face 0 around the tetrahedron centre
         P = c3_pivot + origin_translation.

    Args:
        origin_translation: [3] array/tensor — direction is C3/face-normal axis,
            magnitude is distance from face centre to tetrahedron centre.
        face_offset_radius: float — C3 pivot distance perpendicular to axis.
        device, dtype: torch device and dtype for output tensors.
        coordinate_scaling: scales both ``origin_translation`` and
            ``face_offset_radius`` to match the rigid's unit convention.

    Returns:
        rotations: [12, 3, 3] combined rotation per element (identity first).
        offsets:   [12, 3]   translation offset per element.
    """
    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    face_offset_radius = face_offset_radius * coordinate_scaling
    R_dist = torch.linalg.norm(ot).item()

    # Main axis = direction of origin_translation (default z if zero).
    if R_dist < 1e-8:
        main_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        main_axis = ot / R_dist

    perp1 = _choose_perpendicular(main_axis)
    perp2 = torch.linalg.cross(main_axis, perp1)

    c3_pivot = face_offset_radius * perp1          # face-0 centroid
    P = c3_pivot + ot                               # tetrahedron centre

    I3 = torch.eye(3, device=device, dtype=dtype)

    # --- Intra-face C3 rotations around main_axis ---
    c3_rots = [_rodrigues_matrix(main_axis, 2 * math.pi * i / 3) for i in range(3)]

    # --- Face-to-face rotations (C2 coset representatives) ---
    # The face-to-face map must be a proper tetrahedral C2 element (180° rotation),
    # not an arbitrary rotation by arccos(-1/3).  Both map d0 → d1, but only the
    # C2 preserves the intra-face C3 orientation across faces.
    #
    # C2 axis = bisector of d0 and d1:  k = normalize(d0 + d1).
    # C2 = 2·k·kᵀ − I  (Rodrigues at 180°: cos π = −1, sin π = 0).
    d0 = -main_axis
    d1 = (2.0 * math.sqrt(2) / 3.0) * perp1 + (1.0 / 3.0) * main_axis
    c2_axis = d0 + d1
    c2_axis = c2_axis / torch.linalg.norm(c2_axis)
    C2_face_1 = 2.0 * torch.outer(c2_axis, c2_axis) - I3

    # Faces 2, 3: conjugate C2_face_1 by C3 rotations around main_axis.
    face_rots = [
        I3,                                        # face 0 — identity
        C2_face_1,                                 # face 1
        c3_rots[1] @ C2_face_1 @ c3_rots[2],      # face 2 — C3₁₂₀ C2 C3₂₄₀
        c3_rots[2] @ C2_face_1 @ c3_rots[1],      # face 3 — C3₂₄₀ C2 C3₁₂₀
    ]

    # --- Assemble 12 elements: 4 faces × 3 C3 copies ---
    rotations = []
    offsets = []
    for face_k in range(4):
        for c3_i in range(3):
            R_combined = face_rots[face_k] @ c3_rots[c3_i]
            rotations.append(R_combined)

            # Offset formula (derived from the two-step pivot construction):
            #   δ = R_face_k @ (I − C3_i) @ c3_pivot + (I − R_face_k) @ P
            c3_off = (I3 - c3_rots[c3_i]) @ c3_pivot
            offset = face_rots[face_k] @ c3_off + (I3 - face_rots[face_k]) @ P
            offsets.append(offset)

    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# Octahedral symmetry (24 = 6 faces × C4)
# ---------------------------------------------------------------------------

def _octahedral_transforms(origin_translation, face_offset_radius, device, dtype,
                           coordinate_scaling: float = 1.0):
    """Compute rotation matrices and offsets for octahedral symmetry.

    6 cube faces, each with C4 intra-face symmetry.
    Face normals: ±main_axis, ±perp1, ±perp2.
    """
    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    face_offset_radius = face_offset_radius * coordinate_scaling
    R_dist = torch.linalg.norm(ot).item()

    if R_dist < 1e-8:
        main_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        main_axis = ot / R_dist

    perp1 = _choose_perpendicular(main_axis)
    perp2 = torch.linalg.cross(main_axis, perp1)

    cn_pivot = face_offset_radius * perp1
    P = cn_pivot + ot
    I3 = torch.eye(3, device=device, dtype=dtype)

    # Intra-face C4 around main_axis
    cn_rots = [_rodrigues_matrix(main_axis, math.pi / 2.0 * i) for i in range(4)]

    # Face normals
    d0 = -main_axis                   # base face
    d2 = perp1                        # first adjacent face

    # --- Coset reps (C2 elements) ---
    # Adjacent face d2: bisector C2
    c2_axis = d0 + d2
    c2_axis = c2_axis / torch.linalg.norm(c2_axis)
    C2_adj = 2.0 * torch.outer(c2_axis, c2_axis) - I3

    # Opposite face d1 = +main_axis: 180° around perp1
    C2_opp = 2.0 * torch.outer(perp1, perp1) - I3

    # 4 adjacent faces via C4 conjugation of C2_adj
    face_rots = [
        I3,                                        # face 0  d0 = −main_axis
        C2_opp,                                    # face 1  d1 = +main_axis
        C2_adj,                                    # face 2  d2 = +perp1
        cn_rots[2] @ C2_adj @ cn_rots[2],          # face 3  d3 = −perp1
        cn_rots[1] @ C2_adj @ cn_rots[3],          # face 4  d4 = +perp2
        cn_rots[3] @ C2_adj @ cn_rots[1],          # face 5  d5 = −perp2
    ]

    # Assemble 24 elements: 6 faces × 4 C4 copies
    rotations = []
    offsets = []
    for face_k in range(6):
        for cn_i in range(4):
            rotations.append(face_rots[face_k] @ cn_rots[cn_i])
            cn_off = (I3 - cn_rots[cn_i]) @ cn_pivot
            offsets.append(face_rots[face_k] @ cn_off + (I3 - face_rots[face_k]) @ P)

    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# Icosahedral symmetry (60 = 12 faces × C5)
# ---------------------------------------------------------------------------

def _icosahedron_face_normals(main_axis, perp1, perp2):
    """Return the 12 face normals of a dodecahedron (= icosahedron vertices).

    Layout: south pole, north pole, 5 southern ring, 5 northern ring.
    Southern ring at azimuths 0,72,144,216,288 deg.
    Northern ring at azimuths 36,108,180,252,324 deg (36° stagger).
    """
    r = 2.0 / math.sqrt(5)     # xy-radius of both rings
    z_s = -1.0 / math.sqrt(5)  # z of southern ring
    z_n = 1.0 / math.sqrt(5)   # z of northern ring

    normals = [-main_axis, main_axis]  # d0 = south pole, d1 = north pole

    for i in range(5):  # southern ring (d2 .. d6)
        az = 2.0 * math.pi * i / 5.0
        normals.append(
            r * (math.cos(az) * perp1 + math.sin(az) * perp2) + z_s * main_axis)

    for i in range(5):  # northern ring (d7 .. d11)
        az = 2.0 * math.pi * i / 5.0 + math.radians(36.0)
        normals.append(
            r * (math.cos(az) * perp1 + math.sin(az) * perp2) + z_n * main_axis)

    return normals


def _bisector_c2(a, b, I3):
    """C2 rotation (180°) around the axis bisecting unit vectors a and b.

    This is the edge-midpoint C2 that swaps a ↔ b.
    Formula: C2 = 2·k·kᵀ − I  where k = normalize(a + b).
    """
    k = a + b
    k = k / torch.linalg.norm(k)
    return 2.0 * torch.outer(k, k) - I3


def _icosahedral_transforms(origin_translation, face_offset_radius, device, dtype,
                            coordinate_scaling: float = 1.0):
    """Compute rotation matrices and offsets for icosahedral symmetry.

    12 dodecahedron faces, each with C5 intra-face symmetry.

    Coset reps are built from edge-midpoint C2 elements:
      - Southern ring: C2(d0, d_south), C5-conjugated for the other 4.
      - Northern ring: C2(d_south, d_north) @ C2(d0, d_south), C5-conjugated.
        Two-hop: d0 → d_south → d_north.
      - Opposite pole: C2(d_north, d1) @ C2(d_south, d_north) @ C2(d0, d_south).
        Three-hop: d0 → d_south → d_north → d1.
    """
    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    face_offset_radius = face_offset_radius * coordinate_scaling
    R_dist = torch.linalg.norm(ot).item()

    if R_dist < 1e-8:
        main_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        main_axis = ot / R_dist

    perp1 = _choose_perpendicular(main_axis)
    perp2 = torch.linalg.cross(main_axis, perp1)

    cn_pivot = face_offset_radius * perp1
    P = cn_pivot + ot
    I3 = torch.eye(3, device=device, dtype=dtype)

    # Intra-face C5 around main_axis
    cn_rots = [_rodrigues_matrix(main_axis, 2.0 * math.pi * i / 5.0) for i in range(5)]

    face_normals = _icosahedron_face_normals(main_axis, perp1, perp2)
    d0 = face_normals[0]       # south pole
    d_south = face_normals[2]  # first southern ring vertex  (adjacent to d0)
    d_north = face_normals[7]  # first northern ring vertex  (adjacent to d_south)
    d1 = face_normals[1]       # north pole                  (adjacent to d_north)

    # --- Edge-midpoint C2 building blocks ---
    C2_s = _bisector_c2(d0, d_south, I3)       # d0 ↔ d_south
    C2_sn = _bisector_c2(d_south, d_north, I3)  # d_south ↔ d_north
    C2_n1 = _bisector_c2(d_north, d1, I3)       # d_north ↔ d1

    # --- Coset reps ---
    # Southern ring seed:  d0 → d_south
    rep_south = C2_s
    # Northern ring seed:  d0 → d_south → d_north
    rep_north = C2_sn @ C2_s
    # Opposite pole:       d0 → d_south → d_north → d1
    rep_opp = C2_n1 @ C2_sn @ C2_s

    # Build all 12 face_rots via C5 conjugation
    face_rots = [I3, rep_opp]  # face 0 (south pole), face 1 (north pole)

    for i in range(5):  # southern ring (faces 2..6)
        inv_i = (5 - i) % 5
        face_rots.append(cn_rots[i] @ rep_south @ cn_rots[inv_i])

    for i in range(5):  # northern ring (faces 7..11)
        inv_i = (5 - i) % 5
        face_rots.append(cn_rots[i] @ rep_north @ cn_rots[inv_i])

    # Assemble 60 elements: 12 faces × 5 C5 copies
    rotations = []
    offsets = []
    for face_k in range(12):
        for cn_i in range(5):
            rotations.append(face_rots[face_k] @ cn_rots[cn_i])
            cn_off = (I3 - cn_rots[cn_i]) @ cn_pivot
            offsets.append(face_rots[face_k] @ cn_off + (I3 - face_rots[face_k]) @ P)

    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def _apply_group_symmetry(curr_rigids, symmetry_type, origin_translation,
                          face_offset_radius, coordinate_scaling: float = 1.0):
    """Apply polyhedral symmetry with blow-apart offsets.

    ``coordinate_scaling`` scales the pivot and face offset to match the rigid's
    unit convention (1.0 for unscaled Angstroms, ``ipa_conf.coordinate_scaling``
    for scaled rigids).
    """
    device = curr_rigids._rots.device
    dtype = curr_rigids._trans.dtype

    if symmetry_type == "T":
        rotations, offsets = _tetrahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, coordinate_scaling)
    elif symmetry_type == "O":
        rotations, offsets = _octahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, coordinate_scaling)
    elif symmetry_type == "I":
        rotations, offsets = _icosahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, coordinate_scaling)
    else:
        raise NotImplementedError(
            f"Polyhedral symmetry '{symmetry_type}' is not yet implemented.")

    trans = curr_rigids._trans                   # [B, N, 3]
    rot_mats = curr_rigids._rots.get_rot_mats()  # [B, N, 3, 3]
    B, N = trans.shape[:2]
    G = rotations.shape[0]

    # new_trans[b,g,n,k] = Σ_i  R[g,k,i] · t[b,n,i]  +  δ[g,k]
    new_trans = (
        torch.einsum("gki,bni->bgnk", rotations, trans)
        + offsets[None, :, None, :]
    )
    # new_rots[b,g,n,k,l] = Σ_i  R[g,k,i] · rots[b,n,i,l]
    new_rots = torch.einsum("gki,bnil->bgnkl", rotations, rot_mats)

    return Rigid(
        Rotation(rot_mats=new_rots.reshape(B, G * N, 3, 3)),
        new_trans.reshape(B, G * N, 3),
    )


def apply_symmetry_general(
    curr_rigids,
    symmetry_type,
    symm_n=None,
    origin_translation=np.array([0, 0, 0]),
    face_offset_radius=0.0,
    coordinate_scaling: float = 1.0,
):
    """Unified symmetry expansion for all supported groups.

    For cyclic (C): origin_translation is the pivot for radial offset.
    For T/O/I: two-step construction with face blow-apart and intra-face C3 offset.

    ``coordinate_scaling`` scales ``origin_translation`` and ``face_offset_radius``
    to match the rigid's unit convention. Pass 1.0 (default) for unscaled rigids
    (Angstroms), or ``ipa_conf.coordinate_scaling`` when operating on rigids that
    have been passed through ``scale_rigids``.
    """
    if symmetry_type == "C":
        return apply_symmetry(curr_rigids, symm_n, origin_translation, coordinate_scaling)
    return _apply_group_symmetry(
        curr_rigids, symmetry_type, origin_translation, face_offset_radius, coordinate_scaling)
