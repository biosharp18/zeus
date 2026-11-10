import math
from openfold.utils.rigid_utils import Rigid, Rotation
import torch
import numpy as np


def build_group_permutations(sym_rots):
    """Build the g⁻¹h permutation table for a finite rotation group.

    Given S rotation matrices [R_0, R_1, ..., R_{S-1}] (with R_0 = identity),
    returns an (S, S) integer tensor where entry [g, h] is the index k such
    that R_k = R_g⁻¹ @ R_h. Identity must be at index 0; ordering must match
    the chain layout produced by `_apply_transforms` (chain g lives at slice
    [g*L_asu : (g+1)*L_asu] in the assembled L tensor).

    Also returns the Frobenius gap between the best and second-best match for
    each (g, h), so the caller can sanity-check numerical reliability.

    Args:
        sym_rots: list or stacked tensor of S rotation matrices, each (3, 3).

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
            # Singleton group: no second-best to compare against → use +inf
            # so the "min margin" diagnostic is harmless.
            margins[g, h] = (
                diffs[sorted_idx[1]] - diffs[sorted_idx[0]] if S > 1 else np.inf
            )
    return torch.from_numpy(perm_table).long(), torch.from_numpy(margins).float()


def symmetric_block_tile(x, perm_table):
    """Tile a block-row (B, LR, LC, D...) into a full pair matrix (B, LC, LC, D...).

    Uses a group permutation table to permute block-columns for each block-row.
    For cyclic groups the table is circulant and this is equivalent to a roll.
    For polyhedral groups each row is a different permutation.

    Args:
        x:          (B, LR, LC, D...) — restricted pair features (block-row 0).
        perm_table: (S, S) LongTensor where entry [g, h] = index of g⁻¹h in
                    the group. S = symm_n, LR = LC // S.
    """
    S = perm_table.shape[0]
    LR = x.shape[1]
    rows = []
    for g in range(S):
        perm = perm_table[g]  # (S,)
        col_idx = (perm[:, None] * LR + torch.arange(LR, device=x.device)[None, :]).reshape(-1)
        rows.append(x[:, :, col_idx])
    return torch.cat(rows, dim=1)


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

    `coordinate_scaling` scales the pivot to match the rigid's unit convention.
    Use 1.0 when `curr_rigids` is unscaled (Å); use `ipa_conf.coordinate_scaling`
    (e.g. 0.1) when `curr_rigids` has been passed through `scale_rigids`.
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


def _single_group_order(symmetry_type, symm_n):
    """Size of a single symmetry group (no hierarchical pairing)."""
    if symmetry_type in ("C", "H", "L", "L3"):
        if symm_n is None:
            raise ValueError(f"{symmetry_type} symmetry requires symm_n.")
        return symm_n
    if symmetry_type == "D":
        if symm_n is None:
            raise ValueError("Dihedral symmetry requires symm_n (the C_n subgroup order).")
        return 2 * symm_n
    return _GROUP_ORDERS[symmetry_type]


def get_symmetry_order(symmetry_type, symm_n=None,
                       inner_symmetry_type=None, inner_symm_n=None):
    """Return the total number of group elements.

    When `inner_symmetry_type` is provided, returns the product of outer
    and inner group orders (hierarchical pairing produces G_outer × G_inner
    copies via `_paired_transforms`).
    """
    outer = _single_group_order(symmetry_type, symm_n)
    if inner_symmetry_type is None:
        return outer
    inner = _single_group_order(inner_symmetry_type, inner_symm_n)
    return outer * inner


def _signed_indices(n: int):
    """Signed subunit-centered index list of length n.

    Returns [0, +1, -1, +2, -2, …] truncated to length n. Copy 0 is always
    the identity (so downstream transforms have an identity at index 0,
    which the EfficientIPA path requires). For odd n the result is
    symmetric around zero; for even n there is one extra index on the
    negative side.
    """
    if n < 1:
        raise ValueError(f"signed indices require n >= 1, got {n}")
    n_up = (n - 1) // 2
    n_down = (n - 1) - n_up
    return [0] + list(range(1, n_up + 1)) + list(range(-1, -n_down - 1, -1))


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
        coordinate_scaling: scales both `origin_translation` and
            `face_offset_radius` to match the rigid's unit convention.

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
# Dihedral symmetry (2n = n C_n × C2 flip)
# ---------------------------------------------------------------------------

def _dihedral_transforms(symm_n, origin_translation, ring_offset,
                         device, dtype, coordinate_scaling: float = 1.0):
    """Compute rotation matrices and offsets for dihedral D_n symmetry.

    Two rings of n subunits each, related by a C2 flip about the x-axis.
    The first ring lies at z=0 (in subunit frame); the flipped ring is
    translated by `ring_offset` along z so the two rings don't overlap.

    Convention:
      - C_n rotation axis: z-axis through origin_translation (like cyclic).
      - C2 flip: F = diag(1, -1, -1) (180° about x-axis through origin).
      - Flipped ring z-translation: [0, 0, ring_offset].

    Group action (valid D_n): for g = r^i, T_g(x) = R_i·(x - p) + p with
    R_i = rot_z(2πi/n), p = origin_translation. For g = s·r^j (j = i - n),
    T_g(x) = F·(R_j·(x - p) + p) + [0, 0, ring_offset].

    Closure T_{gh}(x) = T_g(T_h(x)) holds because p lies on the x-axis, so
    F·p = p and (I - R_i)·p has zero z-component, commuting with the ring
    z-shift under left-multiplication by F.

    Args:
        symm_n: total element count (= 2 * n where n is the C_n subgroup
            order). Must be even. This matches the cyclic convention where
            `symm_n` is the number of copies produced.
        origin_translation: [3] pivot for C_n rotation (must lie on x-axis
            for group closure; [r, 0, 0] is the standard choice).
        ring_offset: float z-translation applied to the flipped ring (Å).
        device, dtype: torch device and dtype for output tensors.
        coordinate_scaling: scales `origin_translation` and `ring_offset`
            to match the rigid's unit convention.

    Returns:
        rotations: [symm_n, 3, 3] combined rotation per element (identity first).
        offsets:   [symm_n, 3]    translation offset per element.
    """
    if symm_n % 2 != 0:
        raise ValueError(f"Dihedral symm_n must be even (2*n), got {symm_n}.")
    n = symm_n // 2

    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    ring_offset = float(ring_offset) * coordinate_scaling

    I3 = torch.eye(3, device=device, dtype=dtype)
    F = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        device=device, dtype=dtype,
    )
    z_shift = torch.tensor([0.0, 0.0, ring_offset], device=device, dtype=dtype)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # Intra-ring C_n rotations about z.
    cn_rots = [_rodrigues_matrix(z_axis, 2.0 * math.pi * i / n) for i in range(n)]

    rotations = []
    offsets = []
    # Ring 0: pure C_n around origin_translation (matches cyclic convention).
    for i in range(n):
        R = cn_rots[i]
        rotations.append(R)
        offsets.append((I3 - R) @ ot)
    # Ring 1: C2 flip composed with C_n, plus z-shift to separate the rings.
    for i in range(n):
        R = F @ cn_rots[i]
        rotations.append(R)
        offsets.append(F @ ((I3 - cn_rots[i]) @ ot) + z_shift)

    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# Helical screw symmetry (open orbit, not a group)
# ---------------------------------------------------------------------------

def _helical_transforms(symm_n, twist, rise, origin_translation,
                        device, dtype, coordinate_scaling: float = 1.0,
                        n_start: int = 1):
    """Compute rotation matrices and offsets for a (multi-start) helical screw.

    Single-start (n_start=1): symm_n copies along one screw.
    Multi-start (n_start>1): n_start strands wrapped around the tube axis,
    each with L_per_strand = symm_n / n_start copies via screw. The strands
    are related by a C_{n_start} rotation about the tube axis (through
    `origin_translation`).

    Screw generator (per strand): rotate by `twist` rad about the z-axis
    through `origin_translation`, then translate by `rise` Å along z.
    Strand offset (azimuthal): rotate strand i_s by 2π·i_s / n_start.

    Subunit-centered indexing:
      Index 0 is the identity (so the first L_asu residues in the tensor
      remain the un-transformed subunit — required for the EfficientIPA
      path where keys of the first copy must equal the query subunit).
      Remaining copies distribute symmetrically in both axial and
      azimuthal directions using signed-index lists:
        k_values  = [0, +1, -1, +2, -2, …] length L_per_strand
        i_s_values = [0, +1, -1, +2, -2, …] length n_start
      Flatten order: outer loop over k, inner loop over i_s. This puts
      the subunit's azimuthal (same-z-layer) neighbors immediately after
      the subunit in the tensor, then the axial neighbors.

    Copy placement for offsets (i_s, k):
      α_{i_s} = 2π · i_s / n_start
      R = rot_z(α_{i_s} + k · twist)
      δ = (I − R) · origin_translation + [0, 0, k · rise]
      effective position: R · x + δ.

    Not a closed group: open orbit, composition of two screws can land
    outside the orbit. Fine for FrameDiff (which only needs the orbit).

    Args:
        symm_n: total number of copies G = n_start · L_per_strand.
        twist: rotation per axial step, in radians (NOT scaled).
        rise: translation per axial step along z, in Å.
        origin_translation: [3] offset of the tube axis from the global
            z-axis (e.g. [r, 0, 0] for tube radius r).
        device, dtype: torch device and dtype.
        coordinate_scaling: scales origin_translation and rise; twist and
            azimuthal offset are angles and left unscaled.
        n_start: number of strands around the tube axis (default 1 =
            single-start). Must divide symm_n.

    Returns:
        rotations: [symm_n, 3, 3] per-copy rotation (identity at index 0).
        offsets:   [symm_n, 3]    per-copy translation (zero at index 0).
    """
    G = int(symm_n)
    n_start = int(n_start)
    if G < 1:
        raise ValueError(f"Helical symm_n must be >= 1, got {G}.")
    if n_start < 1:
        raise ValueError(f"Helical n_start must be >= 1, got {n_start}.")
    if G % n_start != 0:
        raise ValueError(
            f"Helical symm_n={G} must be divisible by n_start={n_start}.")
    L_per_strand = G // n_start

    # Signed axial indexing along each strand (subunit at k=0).
    L_up = (L_per_strand - 1) // 2
    L_down = (L_per_strand - 1) - L_up
    k_values = [0] + list(range(1, L_up + 1)) + list(range(-1, -L_down - 1, -1))
    assert len(k_values) == L_per_strand

    # Signed azimuthal indexing across strands (subunit on strand i_s=0).
    n_up = (n_start - 1) // 2
    n_down = (n_start - 1) - n_up
    i_s_values = [0] + list(range(1, n_up + 1)) + list(range(-1, -n_down - 1, -1))
    assert len(i_s_values) == n_start

    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    rise_scaled = float(rise) * coordinate_scaling
    twist = float(twist)

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)

    rotations = []
    offsets = []
    # Outer k, inner i_s — puts azimuthal neighbours of the subunit directly
    # after the subunit in the tensor, followed by axial neighbours.
    for k in k_values:
        z_step = torch.tensor([0.0, 0.0, k * rise_scaled], device=device, dtype=dtype)
        for i_s in i_s_values:
            alpha = 2.0 * math.pi * i_s / n_start
            R = _rodrigues_matrix(z_axis, alpha + k * twist)
            rotations.append(R)
            offsets.append((I3 - R) @ ot + z_step)

    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# 2D lattice / wallpaper symmetry (open orbit, truncated lattice)
# ---------------------------------------------------------------------------

_LATTICE_ALLOWED_POINT_GROUPS = (1, 2, 3, 4, 6)


def _lattice_transforms(symm_n, lattice_a, lattice_b, n_a, n_b,
                        point_group_order, origin_translation,
                        device, dtype, coordinate_scaling: float = 1.0):
    """Compute rotation matrices and offsets for a 2D lattice symmetry.

    Places the subunit at a finite window of the 2D lattice spanned by
    `lattice_a` and `lattice_b`, with an optional intra-site cyclic
    point-group rotation of order `point_group_order` about the z-axis
    through `origin_translation`. Covers wallpaper groups p1 (point_group=1),
    p2, p3, p4, p6 and degenerate 1D frieze (n_b=1).

    Transform for copy (m, n, g), with signed subunit-centered indexing:
      R = rot_z(2π · g / point_group_order)
      δ = (I − R) · origin_translation + m · a + n · b
    Copy 0 = (m=0, n=0, g=0) = identity (required by EfficientIPA path).

    Flatten order (outer m, middle n, inner g) puts the point-group ring
    at the subunit's own site immediately after the subunit in the tensor,
    then lattice neighbors.

    Open orbit: truncated lattice window. Not a closed group; fine for
    FrameDiff which only needs the orbit, not closure.

    Args:
        symm_n: total copies G = n_a · n_b · point_group_order.
        lattice_a, lattice_b: [3] basis vectors in Å. Typically in xy-plane;
            the point-group rotation is about z and will only give a sensible
            ring if the basis vectors lie in the xy-plane (required when
            point_group_order > 1).
        n_a, n_b: number of unit cells along each basis direction.
        point_group_order: intra-site C_k, k ∈ {1, 2, 3, 4, 6}.
        origin_translation: [3] pivot for the intra-site point-group rotation.
        device, dtype: torch device and dtype for outputs.
        coordinate_scaling: scales lattice_a, lattice_b, origin_translation
            to match the rigid's unit convention.

    Returns:
        rotations: [symm_n, 3, 3] per-copy rotation (identity at index 0).
        offsets:   [symm_n, 3]    per-copy translation (zero at index 0).
    """
    G = int(symm_n)
    p = int(point_group_order)
    n_a = int(n_a)
    n_b = int(n_b)
    if p not in _LATTICE_ALLOWED_POINT_GROUPS:
        raise ValueError(
            f"Lattice point_group_order must be in {_LATTICE_ALLOWED_POINT_GROUPS}, got {p}.")
    if n_a < 1 or n_b < 1:
        raise ValueError(f"Lattice n_a={n_a}, n_b={n_b} must be >= 1.")
    if G != n_a * n_b * p:
        raise ValueError(
            f"Lattice symm_n={G} must equal n_a({n_a}) * n_b({n_b}) "
            f"* point_group_order({p}) = {n_a * n_b * p}.")

    a = torch.as_tensor(lattice_a, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    b = torch.as_tensor(lattice_b, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)

    m_vals = _signed_indices(n_a)
    n_vals = _signed_indices(n_b)
    g_vals = _signed_indices(p)

    # Pre-build C_p rotations (indexed by raw g value so 0 → identity).
    cn_rots = {g: _rodrigues_matrix(z_axis, 2.0 * math.pi * g / p) for g in g_vals}

    rotations = []
    offsets = []
    # Order: outer m, middle n, inner g. (0, 0, 0) lands at index 0.
    for m in m_vals:
        for n in n_vals:
            lattice_offset = m * a + n * b
            for g in g_vals:
                R = cn_rots[g]
                rotations.append(R)
                offsets.append((I3 - R) @ ot + lattice_offset)

    return torch.stack(rotations), torch.stack(offsets)


def _lattice3d_transforms(symm_n, lattice_a, lattice_b, lattice_c,
                          n_a, n_b, n_c, device, dtype,
                          coordinate_scaling: float = 1.0):
    """Compute rotation matrices and offsets for a 3D lattice symmetry.

    Places the subunit at a finite n_a × n_b × n_c window of the 3D lattice
    spanned by `lattice_a`, `lattice_b`, `lattice_c`. All copies are pure
    translations of the subunit (rotations are identity); the lattice
    geometry (cubic, hex, FCC, BCC, etc.) is encoded entirely by the choice
    of basis vectors.

    Transform for copy (m, n, p):
      R = I
      δ = m · a + n · b + p · c
    Copy 0 = (m=0, n=0, p=0) = identity (required by EfficientIPA path).

    Subunit-centered indexing (`_signed_indices`) keeps copy 0 at index 0
    and arranges remaining copies symmetrically around the origin so the
    overall lattice tile is centered.

    Open orbit: truncated lattice window, not a closed group; that's fine
    for FrameDiff which only needs the orbit, not closure.

    Args:
        symm_n: total copies G = n_a · n_b · n_c.
        lattice_a, lattice_b, lattice_c: [3] basis vectors in Å.
        n_a, n_b, n_c: number of unit cells along each basis direction.
        device, dtype: torch device and dtype for outputs.
        coordinate_scaling: scales all three basis vectors to match the
            rigid's unit convention.

    Returns:
        rotations: [symm_n, 3, 3] per-copy rotation (all identity).
        offsets:   [symm_n, 3]    per-copy translation (zero at index 0).
    """
    G = int(symm_n)
    n_a = int(n_a)
    n_b = int(n_b)
    n_c = int(n_c)
    if n_a < 1 or n_b < 1 or n_c < 1:
        raise ValueError(
            f"Lattice3D n_a={n_a}, n_b={n_b}, n_c={n_c} must be >= 1.")
    if G != n_a * n_b * n_c:
        raise ValueError(
            f"Lattice3D symm_n={G} must equal "
            f"n_a({n_a}) * n_b({n_b}) * n_c({n_c}) = {n_a * n_b * n_c}.")

    a = torch.as_tensor(lattice_a, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    b = torch.as_tensor(lattice_b, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    c = torch.as_tensor(lattice_c, device=device, dtype=dtype).clone().detach() * coordinate_scaling

    I3 = torch.eye(3, device=device, dtype=dtype)

    m_vals = _signed_indices(n_a)
    n_vals = _signed_indices(n_b)
    p_vals = _signed_indices(n_c)

    rotations = []
    offsets = []
    # Order: outer m, middle n, inner p. (0, 0, 0) lands at index 0.
    for m in m_vals:
        for n in n_vals:
            for p in p_vals:
                rotations.append(I3)
                offsets.append(m * a + n * b + p * c)

    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# Cyclic transforms exposed as (R, δ) — used by the hierarchical inner path.
# ---------------------------------------------------------------------------

def _cyclic_transforms(symm_n, origin_translation, device, dtype,
                       coordinate_scaling: float = 1.0):
    """Cyclic C_n transforms as explicit (R, δ) tensors.

    Matches `apply_symmetry` semantics: R_i = rot_z(2πi/n), δ_i = (I - R_i)·p,
    with sequential indexing [0, 1, …, n-1]. Identity is at index 0.
    """
    n = int(symm_n)
    if n < 1:
        raise ValueError(f"Cyclic symm_n must be >= 1, got {n}.")
    ot = torch.as_tensor(origin_translation, device=device, dtype=dtype).clone().detach() * coordinate_scaling
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)
    rotations = []
    offsets = []
    for i in range(n):
        R = _rodrigues_matrix(z_axis, 2.0 * math.pi * i / n)
        rotations.append(R)
        offsets.append((I3 - R) @ ot)
    return torch.stack(rotations), torch.stack(offsets)


# ---------------------------------------------------------------------------
# Hierarchical pairing: compose two symmetries as a Cartesian product.
# ---------------------------------------------------------------------------

def _paired_transforms(R_outer, d_outer, R_inner, d_inner):
    """Compose outer and inner symmetries as a Cartesian product action.

    For point x, the composed action applies inner first, then outer:
      x → R_outer · (R_inner · x + d_inner) + d_outer
        = (R_outer · R_inner) · x + (R_outer · d_inner + d_outer)

    Layout: outer index slow, inner index fast. Index 0 = (outer[0], inner[0]).
    If both inputs have identity at index 0 (our convention throughout this
    module), the paired identity is also at index 0 — required by the
    EfficientIPA path which assumes keys[:n_subunit] = queries.

    Args:
        R_outer: [G_o, 3, 3] outer rotations.
        d_outer: [G_o, 3]   outer offsets.
        R_inner: [G_i, 3, 3] inner rotations.
        d_inner: [G_i, 3]   inner offsets.

    Returns:
        R_pair: [G_o * G_i, 3, 3]
        d_pair: [G_o * G_i, 3]
    """
    G_o = R_outer.shape[0]
    G_i = R_inner.shape[0]
    # R_pair[o, i] = R_outer[o] @ R_inner[i]
    R_pair = torch.einsum("oij,pjk->opik", R_outer, R_inner).reshape(G_o * G_i, 3, 3)
    # d_pair[o, i] = R_outer[o] @ d_inner[i] + d_outer[o]
    d_pair = (torch.einsum("oij,pj->opi", R_outer, d_inner)
              + d_outer[:, None, :]).reshape(G_o * G_i, 3)
    return R_pair, d_pair


# ---------------------------------------------------------------------------
# Generic dispatch that returns (R, δ) for any supported symmetry type.
# Used only by the hierarchical (outer × inner) path. The single-symmetry
# fast paths remain unchanged.
# ---------------------------------------------------------------------------

def _get_symmetry_transforms(
    symmetry_type, symm_n, origin_translation,
    face_offset_radius, ring_offset, twist, rise, n_start,
    lattice_a, lattice_b, n_a, n_b, point_group_order,
    device, dtype, coordinate_scaling: float = 1.0,
    lattice_c=None, n_c=1,
):
    """Dispatch to the appropriate per-type transform builder, returning (R, δ)."""
    if symmetry_type == "C":
        return _cyclic_transforms(symm_n, origin_translation, device, dtype, coordinate_scaling)
    if symmetry_type == "D":
        return _dihedral_transforms(
            symm_n, origin_translation, ring_offset, device, dtype, coordinate_scaling)
    if symmetry_type == "H":
        return _helical_transforms(
            symm_n, twist, rise, origin_translation, device, dtype,
            coordinate_scaling, n_start=n_start)
    if symmetry_type == "L":
        return _lattice_transforms(
            symm_n, lattice_a, lattice_b, n_a, n_b,
            point_group_order, origin_translation,
            device, dtype, coordinate_scaling)
    if symmetry_type == "L3":
        if lattice_c is None:
            raise ValueError("L3 symmetry requires lattice_c.")
        return _lattice3d_transforms(
            symm_n, lattice_a, lattice_b, lattice_c, n_a, n_b, n_c,
            device, dtype, coordinate_scaling)
    if symmetry_type == "T":
        return _tetrahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, coordinate_scaling)
    if symmetry_type == "O":
        return _octahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, coordinate_scaling)
    if symmetry_type == "I":
        return _icosahedral_transforms(
            origin_translation, face_offset_radius, device, dtype, coordinate_scaling)
    raise NotImplementedError(f"Symmetry '{symmetry_type}' not supported.")


def _apply_transforms(curr_rigids, rotations, offsets):
    """Apply per-copy (R, δ) to a Rigid, producing G copies of each residue.

    new_rigid[b, g·N + n] = (R_g · R_0, R_g · t_0 + δ_g).
    """
    trans = curr_rigids._trans                   # [B, N, 3]
    rot_mats = curr_rigids._rots.get_rot_mats()  # [B, N, 3, 3]
    B, N = trans.shape[:2]
    G = rotations.shape[0]
    new_trans = (
        torch.einsum("gki,bni->bgnk", rotations, trans)
        + offsets[None, :, None, :]
    )
    new_rots = torch.einsum("gki,bnil->bgnkl", rotations, rot_mats)
    return Rigid(
        Rotation(rot_mats=new_rots.reshape(B, G * N, 3, 3)),
        new_trans.reshape(B, G * N, 3),
    )


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def _apply_group_symmetry(curr_rigids, symmetry_type, symm_n, origin_translation,
                          face_offset_radius, ring_offset, twist, rise, n_start,
                          lattice_a, lattice_b, n_a, n_b, point_group_order,
                          coordinate_scaling: float = 1.0,
                          lattice_c=None, n_c=1):
    """Apply non-cyclic symmetry (D/T/O/I/H/L/L3) with blow-apart offsets.

    `coordinate_scaling` scales the pivot and distance-valued offsets to
    match the rigid's unit convention (1.0 for unscaled Å,
    `ipa_conf.coordinate_scaling` for scaled rigids). Angle-valued params
    (twist) are left unscaled.
    """
    device = curr_rigids._rots.device
    dtype = curr_rigids._trans.dtype

    if symmetry_type == "D":
        rotations, offsets = _dihedral_transforms(
            symm_n, origin_translation, ring_offset, device, dtype, coordinate_scaling)
    elif symmetry_type == "H":
        rotations, offsets = _helical_transforms(
            symm_n, twist, rise, origin_translation, device, dtype,
            coordinate_scaling, n_start=n_start)
    elif symmetry_type == "L":
        rotations, offsets = _lattice_transforms(
            symm_n, lattice_a, lattice_b, n_a, n_b,
            point_group_order, origin_translation,
            device, dtype, coordinate_scaling)
    elif symmetry_type == "L3":
        if lattice_c is None:
            raise ValueError("L3 symmetry requires lattice_c.")
        rotations, offsets = _lattice3d_transforms(
            symm_n, lattice_a, lattice_b, lattice_c,
            n_a, n_b, n_c, device, dtype, coordinate_scaling)
    elif symmetry_type == "T":
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
            f"Symmetry '{symmetry_type}' is not yet implemented.")

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
    ring_offset=0.0,
    twist=0.0,
    rise=0.0,
    n_start=1,
    lattice_a=np.array([0.0, 0.0, 0.0]),
    lattice_b=np.array([0.0, 0.0, 0.0]),
    lattice_c=np.array([0.0, 0.0, 0.0]),
    n_a=1,
    n_b=1,
    n_c=1,
    point_group_order=1,
    inner_symmetry_type=None,
    inner_symm_n=None,
    inner_origin_translation=np.array([0.0, 0.0, 0.0]),
    coordinate_scaling: float = 1.0,
):
    """Unified symmetry expansion for all supported groups.

    For cyclic (C): origin_translation is the pivot for radial offset.
    For dihedral (D): origin_translation is the C_n pivot (must lie on the
        x-axis for group closure); ring_offset is the z-translation between
        the two rings.
    For helical (H): origin_translation sets the helix radius (off-axis
        offset from the z-axis); twist (rad) and rise (Å) define the screw
        step. Optional `n_start` > 1 produces a multi-start helix (tube/
        cylinder) with n_start strands around the axis; symm_n must be
        divisible by n_start. Subunit-centered indexing places the first
        L_asu at the middle of the helix (see `_helical_transforms`).
    For lattice (L): 2D lattice with lattice_a, lattice_b as basis vectors,
        n_a * n_b unit cells, and an intra-site cyclic point-group rotation
        of order point_group_order (1/2/3/4/6) about the z-axis through
        origin_translation. symm_n must equal n_a * n_b * point_group_order.
        Covers wallpaper groups p1 (p=1), p2, p3, p4, p6; 1D frieze when
        n_b=1.
    For 3D lattice (L3): n_a * n_b * n_c unit cells of a Bravais lattice
        spanned by lattice_a, lattice_b, lattice_c. All copies are pure
        translations of the subunit (no rotation). symm_n must equal
        n_a * n_b * n_c. Use cubic basis (a·êx, a·êy, a·êz) for simple
        cubic, or hexagonal (a·êx, a/2·êx+a√3/2·êy, c·êz) for HCP-style.
    For T/O/I: two-step construction with face blow-apart and intra-face C3 offset.

    `coordinate_scaling` scales all distance-valued parameters
    (origin_translation, face_offset_radius, ring_offset, rise, lattice_a,
    lattice_b) to match the rigid's unit convention. `twist` is an angle
    and left unscaled. Pass 1.0 (default) for unscaled rigids (Å), or
    `ipa_conf.coordinate_scaling` when operating on rigids that have been
    passed through `scale_rigids`.
    """
    if inner_symmetry_type is None:
        # Single-group fast path — bit-identical to pre-hierarchical behavior.
        if symmetry_type == "C":
            return apply_symmetry(curr_rigids, symm_n, origin_translation, coordinate_scaling)
        return _apply_group_symmetry(
            curr_rigids, symmetry_type, symm_n, origin_translation,
            face_offset_radius, ring_offset, twist, rise, n_start,
            lattice_a, lattice_b, n_a, n_b, point_group_order, coordinate_scaling,
            lattice_c=lattice_c, n_c=n_c)

    # Hierarchical path: compose outer × inner as a Cartesian-product action.
    # The inner symmetry is (currently) a cyclic C_n placed in the subunit's
    # local frame — sufficient to express T=n-style capsid quasi-trimers/
    # pentamers/tetramers and other nested clusterings. To support inner
    # H/L/D would plumb more kwargs; v1 keeps the API minimal.
    #
    # Convention: `symm_n` here is the OVERALL total (outer · inner). All
    # downstream consumers (EfficientIPA's symm_s tile, subunit slicing)
    # expect this total, so keep the same meaning as the single-group path.
    # The outer group order is derived by dividing out the inner total.
    inner_total = _single_group_order(inner_symmetry_type, inner_symm_n)
    if symm_n % inner_total != 0:
        raise ValueError(
            f"Hierarchical symm_n={symm_n} must be divisible by inner "
            f"total order ({inner_symmetry_type}: {inner_total}).")
    outer_total = symm_n // inner_total
    device = curr_rigids._rots.device
    dtype = curr_rigids._trans.dtype
    R_o, d_o = _get_symmetry_transforms(
        symmetry_type, outer_total, origin_translation,
        face_offset_radius, ring_offset, twist, rise, n_start,
        lattice_a, lattice_b, n_a, n_b, point_group_order,
        device, dtype, coordinate_scaling,
        lattice_c=lattice_c, n_c=n_c,
    )
    # Inner: reuse the generic dispatcher so other types can be added later.
    # All non-used inner params fall back to defaults.
    R_i, d_i = _get_symmetry_transforms(
        inner_symmetry_type, inner_symm_n, inner_origin_translation,
        face_offset_radius=0.0, ring_offset=0.0, twist=0.0, rise=0.0, n_start=1,
        lattice_a=np.zeros(3), lattice_b=np.zeros(3), n_a=1, n_b=1,
        point_group_order=1,
        device=device, dtype=dtype,
        coordinate_scaling=coordinate_scaling,
        lattice_c=np.zeros(3), n_c=1,
    )
    R_pair, d_pair = _paired_transforms(R_o, d_o, R_i, d_i)
    return _apply_transforms(curr_rigids, R_pair, d_pair)
