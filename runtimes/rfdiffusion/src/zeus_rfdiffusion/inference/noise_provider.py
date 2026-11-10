"""NoiseProvider: dedicated PRNG channel for diffusion noise.

Used so two inference runs (e.g. null-attention vs efficient-attention) can
share bit-exact reverse-diffusion noise even though their model forwards
consume the global RNG differently.

Three impls:
  - FreshNoise:     samples from dedicated np.random.Generator and torch.Generator.
  - RecordingNoise: wraps Fresh; appends every draw to an internal log; dump() to .pt.
  - ReplayNoise:    loads a recorded .pt and returns the matching tensor for each
                    (timestep, kind) pair, asserting shape compatibility.

Two kinds of noise are emitted:
  - 'translation' (torch.Tensor on a target device/dtype)
  - 'rotation'    (np.ndarray, used by IGSO3.reverse_sample_vectorized)

Each call is keyed by (timestep, kind, draw_index) where draw_index is the
nth call within that (t, kind). This lets a single timestep emit multiple
draws if the call site is iterated.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

# Sentinel kind names so callers can't accidentally typo them.
KIND_TRANSLATION = "translation"
KIND_ROTATION = "rotation"
_VALID_KINDS = {KIND_TRANSLATION, KIND_ROTATION}


def _shape_to_tuple(shape) -> Tuple[int, ...]:
    if isinstance(shape, torch.Size):
        return tuple(int(s) for s in shape)
    return tuple(int(s) for s in shape)


class NoiseProvider:
    """Abstract base. Subclasses must implement translation() and rotation()."""

    def translation(self, t: int, shape, *, device=None, dtype=None) -> torch.Tensor:
        raise NotImplementedError

    def rotation(self, t: int, shape) -> np.ndarray:
        raise NotImplementedError

    def begin_step(self, t: int) -> None:
        """Called by the sampler at the top of each reverse step.

        Lets subclasses reset per-step counters."""
        pass

    def reseed(self, seed: int) -> None:
        """Rewind the noise stream to a known point.

        The generators here are private, so the global torch/numpy reseeding
        that run_inference.py does between designs cannot reach them. Without
        an explicit rewind the stream runs on from wherever the previous design
        left it, which makes design i depend on how many designs preceded it.
        """
        pass


class FreshNoise(NoiseProvider):
    """Independent np / torch Generators seeded once at construction.

    Using dedicated generators (rather than the global torch.manual_seed /
    np.random.seed state) means model-forward RNG consumption cannot affect
    the noise we emit here.
    """

    def __init__(self, seed: int = 0):
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self._seed = int(seed)
        self._np_rng = np.random.default_rng(self._seed)
        self._torch_gen = torch.Generator(device="cpu")
        self._torch_gen.manual_seed(self._seed)

    def translation(self, t, shape, *, device=None, dtype=None):
        shape = _shape_to_tuple(shape)
        z = torch.randn(shape, generator=self._torch_gen, dtype=torch.float32)
        if dtype is not None:
            z = z.to(dtype=dtype)
        if device is not None:
            z = z.to(device=device)
        return z

    def rotation(self, t, shape):
        shape = _shape_to_tuple(shape)
        return self._np_rng.standard_normal(size=shape)


class RecordingNoise(NoiseProvider):
    """Wraps FreshNoise; logs every draw keyed by (t, kind, draw_index)."""

    def __init__(self, seed: int = 0):
        self._fresh = FreshNoise(seed=seed)
        # log[(t, kind)] -> list of np.ndarray (CPU, float32 / float64)
        self._log: dict = {}
        self._cur_t = None
        self._draw_counter: dict = {}  # (t, kind) -> int

    def begin_step(self, t):
        self._cur_t = int(t)

    def reseed(self, seed):
        self._fresh.reseed(seed)

    def _key(self, t, kind):
        t = int(t)
        return (t, kind)

    def _append(self, t, kind, arr_np):
        key = self._key(t, kind)
        self._log.setdefault(key, []).append(arr_np)

    def translation(self, t, shape, *, device=None, dtype=None):
        z = self._fresh.translation(t, shape, device=device, dtype=dtype)
        self._append(t, KIND_TRANSLATION, z.detach().cpu().numpy())
        return z

    def rotation(self, t, shape):
        z = self._fresh.rotation(t, shape)
        self._append(t, KIND_ROTATION, np.asarray(z))
        return z

    def dump(self, path: str) -> None:
        # Convert keys to strings for torch.save round-trip.
        serializable = {f"{t}|{kind}": [torch.from_numpy(a) for a in arrs]
                        for (t, kind), arrs in self._log.items()}
        torch.save({"version": 1, "seed": self._fresh._seed, "log": serializable}, path)


class ReplayNoise(NoiseProvider):
    """Reads a recording back. Each (t, kind) maintains its own draw counter."""

    def __init__(self, path: str):
        blob = torch.load(path, map_location="cpu")
        if blob.get("version") != 1:
            raise ValueError(f"Unsupported NoiseRecorder version: {blob.get('version')}")
        # Parse keys back into tuples.
        log = {}
        for k, arrs in blob["log"].items():
            t_str, kind = k.split("|", 1)
            log[(int(t_str), kind)] = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for a in arrs]
        self._log = log
        self._draw_counter: dict = {}
        self._seed = blob.get("seed", None)

    def begin_step(self, t):
        # Reset draw counter for this t so the same recording can be replayed twice.
        for k in list(self._draw_counter.keys()):
            if k[0] == int(t):
                self._draw_counter[k] = 0

    def _next(self, t, kind, expected_shape):
        key = (int(t), kind)
        if key not in self._log:
            raise KeyError(f"No noise recorded for t={t}, kind={kind}")
        idx = self._draw_counter.get(key, 0)
        bucket = self._log[key]
        if idx >= len(bucket):
            raise IndexError(
                f"Replay exhausted for (t={t}, kind={kind}): "
                f"requested draw #{idx} but only {len(bucket)} recorded"
            )
        arr = bucket[idx]
        self._draw_counter[key] = idx + 1
        if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(
                f"Replay shape mismatch at (t={t}, kind={kind}, draw={idx}): "
                f"recorded {tuple(arr.shape)}, requested {tuple(expected_shape)}"
            )
        return arr

    def translation(self, t, shape, *, device=None, dtype=None):
        shape = _shape_to_tuple(shape)
        arr = self._next(t, KIND_TRANSLATION, shape)
        z = torch.from_numpy(arr).to(dtype=dtype if dtype is not None else torch.float32)
        if device is not None:
            z = z.to(device=device)
        return z

    def rotation(self, t, shape):
        shape = _shape_to_tuple(shape)
        return np.asarray(self._next(t, KIND_ROTATION, shape))


def make_provider(mode: str, *, seed: int = 0, path: str = None) -> NoiseProvider:
    """Factory used by the CLI.

    mode:
      - 'off' / 'fresh': FreshNoise (default behavior, independent RNG channel)
      - 'record':        RecordingNoise; caller must call .dump(path) after
      - 'replay':        ReplayNoise(path)
    """
    if mode is None or mode is False:
        mode = "off"
    if isinstance(mode, bool):  # YAML 1.1 quirk: 'off'/'no'/'false' parse as False
        mode = "off"
    mode = str(mode).lower()
    if mode in ("off", "fresh"):
        return FreshNoise(seed=seed)
    if mode == "record":
        if path is None:
            raise ValueError("noise_mode='record' requires noise_path")
        return RecordingNoise(seed=seed)
    if mode == "replay":
        if path is None:
            raise ValueError("noise_mode='replay' requires noise_path")
        return ReplayNoise(path)
    raise ValueError(f"Unknown noise_mode: {mode!r}")
