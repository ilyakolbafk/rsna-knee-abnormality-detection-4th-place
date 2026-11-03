"""Core v2 arm A: the SR-E1 augmentation block re-targeted at the frozen v8.3 token bank (336 px, 140 mm).

Design (see prompts/codex/CORE_V2_ARM_A_SPEC_20260918.md):
- geometry + intensity run on GPU on the 336x336 source tokens, *before* the untouched v8.3 path
  (H1.features: bilinear 336->384, per-study gain, clamp, (x-0.5)/0.5). The warp resamples 336->336 on the
  same pixel grid, so at identity parameters every sampling coordinate is an integer pixel centre and the
  old path is reproduced (the 1-px border deviation a direct 336->384 grid_sample would introduce is avoided);
- one parameter vector per series (semantic slot), noise realisation per token;
- the affine grid is built from elementwise ops, not F.affine_grid: NGC images force cuBLAS TF32
  (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1) and the bmm inside affine_grid then drifts by ~2e-4 normalised units;
- no flips, ever: tokens are canonical_L_v8_3, a flip would swap medial and lateral.
Written by Claude 2026-09-18.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

SRC = 336  # v8.3 token side (140 mm)
MM_PER_PX = 140.0 / SRC

# Parameter vector per token (P = 13), identical layout to small_dense_e1_data.PARAMS:
# 0 side scale k (sqrt of crop area), 1 rotation rad, 2 shift x, 3 shift y (fraction of side),
# 4 gamma, 5 contrast, 6-10 bias-field coefficients (x, y, x2, xy, y2),
# 11 blur sigma in source px (0 = none), 12 gaussian noise std (0 = none).
PARAMS = 13
IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0.0, 0.0], np.float32)

# Frozen ranges (spec section 3). CROP_AREA is the crop *area* fraction, as in E1: side = sqrt(area).
CROP_AREA = (0.85, 1.00)          # side 0.922-1.000 -> 129-140 mm field of view
ROTATION_DEG = 12.0
SHIFT_FRACTION = 0.08             # of the side, i.e. up to 11 mm
GAMMA = (0.7, 1.4)                # log-uniform
CONTRAST = (0.8, 1.2)
BIAS_MAX = 0.10
BLUR_P, BLUR_SIGMA = 0.25, (0.45, 1.5)   # E1 used 0.3-1.0 px at 224 px/140 mm; same physical range at 336 px
BLUR_TAPS = 7                             # +-3 taps cover 2 sigma at sigma 1.5
NOISE_P, NOISE_STD = 0.5, (0.005, 0.03)

SERIES_DROP_P = 0.10
TOKEN_DROP_P = 0.15
POSITION_JITTER = 1


def series_params(rng) -> np.ndarray:
    """One parameter vector for a whole series; `rng` is a seeded `random.Random`."""
    value = IDENTITY.copy()
    value[0] = math.sqrt(rng.uniform(*CROP_AREA))
    value[1] = math.radians(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
    value[2] = rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION)
    value[3] = rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION)
    value[4] = math.exp(rng.uniform(math.log(GAMMA[0]), math.log(GAMMA[1])))
    value[5] = rng.uniform(*CONTRAST)
    value[6:11] = [rng.uniform(-BIAS_MAX, BIAS_MAX) for _ in range(5)]
    value[11] = rng.uniform(*BLUR_SIGMA) if rng.random() < BLUR_P else 0.0
    value[12] = rng.uniform(*NOISE_STD) if rng.random() < NOISE_P else 0.0
    return value


def is_identity(params: torch.Tensor) -> bool:
    return bool(torch.equal(params.float().cpu(), torch.from_numpy(IDENTITY).expand(len(params), -1)))


def sampling_grid(p: torch.Tensor, out: int = SRC) -> torch.Tensor:
    """Same geometry as F.affine_grid(theta, align_corners=False) with
    theta = [[k cos, -k sin, 2 tx], [k sin, k cos, 2 ty]] but built elementwise (exact under forced TF32).
    At identity the normalised coordinate of output pixel i is (2i+1)/out-1, i.e. source pixel i exactly."""
    k, rot, tx, ty = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    cos, sin = torch.cos(rot), torch.sin(rot)
    axis = (torch.arange(out, device=p.device, dtype=p.dtype) * 2 + 1) / out - 1
    gx, gy = axis.view(1, 1, out), axis.view(1, out, 1)
    view = lambda value: value.view(-1, 1, 1)  # noqa: E731
    x = view(k * cos) * gx + view(-k * sin) * gy + view(2 * tx)
    y = view(k * sin) * gx + view(k * cos) * gy + view(2 * ty)
    return torch.stack([x, y], dim=-1)


def _bias_basis(device, dtype, side: int = SRC):
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx, yy, xx * xx - 1.0 / 3.0, xx * yy, yy * yy - 1.0 / 3.0], 0)


def warp(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Geometric part only: [N,3,336,336] -> [N,3,336,336], zeros outside the source (black corners)."""
    return F.grid_sample(x, sampling_grid(params.float(), x.shape[-1]), mode="bilinear", padding_mode="zeros", align_corners=False)


@torch.no_grad()
def augment336(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """x [N,3,336,336] in [0,1]; params [N,13]. Returns [N,3,336,336] in [0,1], float32.

    Batch-level fast path: when every row is IDENTITY the input is returned unchanged (bit-exact old path).
    Per-study gain is *not* applied here; it stays in the untouched H1.features.
    """
    if x.shape[-2:] != (SRC, SRC):
        raise ValueError(f"Expected v8.3 {SRC}x{SRC} tokens, got {tuple(x.shape[-2:])}")
    if params.shape != (x.shape[0], PARAMS):
        raise ValueError(f"params must be [{x.shape[0]},{PARAMS}], got {tuple(params.shape)}")
    if is_identity(params):
        return x
    with torch.autocast(x.device.type, enabled=False):
        x = x.float()
        p = params.float()
        out = warp(x, p)
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = ((out - mean) * p[:, 5].view(-1, 1, 1, 1) + mean).clamp_(0, 1)
        out = out.pow(p[:, 4].view(-1, 1, 1, 1))
        basis = _bias_basis(x.device, out.dtype, x.shape[-1])
        exponent = sum(p[:, 6 + i].view(-1, 1, 1) * basis[i] for i in range(5))
        out = (out * torch.exp(exponent).unsqueeze(1)).clamp_(0, 1)
        sigma = p[:, 11]
        blur = sigma > 0
        if bool(blur.any()):
            xs, s = out[blur], sigma[blur]
            m, side = xs.shape[0], xs.shape[-1]
            half = BLUR_TAPS // 2
            t = torch.arange(-half, half + 1, device=x.device, dtype=out.dtype)
            kern = torch.exp(-(t[None, :] ** 2) / (2 * s[:, None] ** 2))
            kern = (kern / kern.sum(1, keepdim=True)).repeat_interleave(3, 0)
            z = xs.reshape(1, m * 3, side, side)
            z = F.conv2d(F.pad(z, (half, half, 0, 0), mode="reflect"), kern.view(m * 3, 1, 1, BLUR_TAPS), groups=m * 3)
            z = F.conv2d(F.pad(z, (0, 0, half, half), mode="reflect"), kern.view(m * 3, 1, BLUR_TAPS, 1), groups=m * 3)
            out[blur] = z.view(m, 3, side, side)
        noise = p[:, 12].view(-1, 1, 1, 1)
        if bool((noise > 0).any()):
            out = out + torch.randn_like(out) * noise
        return out.clamp_(0, 1)


# ----------------------------------------------------------------------------------------------------------
# Token-level operations on the compact bank (refs into `bank`), all CPU/numpy, all deterministic per rng.


def series_drop_mask(slots_present, rng, p: float = SERIES_DROP_P) -> np.ndarray:
    """Keep mask over the given slots; at least one slot always survives."""
    slots_present = np.asarray(slots_present, np.int64)
    keep = np.asarray([rng.random() >= p for _ in slots_present], bool)
    if len(keep) and not keep.any():
        keep[rng.randrange(len(keep))] = True
    return keep


def token_keep_mask(count: int, rng, p: float = TOKEN_DROP_P) -> np.ndarray:
    """Bernoulli(1-p) keep mask over the sampled windows; at least one window always survives."""
    keep = np.asarray([rng.random() >= p for _ in range(count)], bool)
    if count and not keep.any():
        keep[rng.randrange(count)] = True
    return keep


def neighbor_maps(refs: np.ndarray, valid: np.ndarray, zero_frame: np.ndarray):
    """Successor/predecessor bank-frame maps in *stored channel order*, from provably adjacent channels.

    A channel is a real neighbour only when it differs from the centre ref (the builder replicates the centre
    at series ends and decode gaps) and is not the all-zero plane (geometry-unverified channels). Shared
    frames never cross series, so the maps never link two series. Contradictory links (should not exist)
    are dropped fail-closed.
    """
    succ, pred, bad = {}, {}, set()

    def link(a, b):
        if succ.get(a, b) != b or pred.get(b, a) != a:
            bad.update((a, b))
        succ[a] = b
        pred[b] = a

    for t in np.flatnonzero(valid):
        a, b, c = (int(v) for v in refs[t])
        if a != b and not zero_frame[a]:
            link(a, b)
        if c != b and not zero_frame[c]:
            link(b, c)
    for key in bad:
        succ.pop(key, None)
        pred.pop(key, None)
    return succ, pred


def jitter_refs(refs: np.ndarray, valid: np.ndarray, zero_frame: np.ndarray, chosen: np.ndarray, rng, amount: int = POSITION_JITTER):
    """Return (refs[len(chosen),3], applied[len(chosen)]) with each chosen window moved by -1/0/+1 slice in
    stored channel order when the required frames exist in the bank; otherwise the window is left as is."""
    succ, pred = neighbor_maps(refs, valid, zero_frame)
    out = refs[chosen].astype(np.int64).copy()
    applied = np.zeros(len(chosen), bool)
    if amount <= 0:
        return out, applied
    for i, token in enumerate(chosen):
        shift = rng.randint(-amount, amount)
        a, b, c = (int(v) for v in refs[token])
        if shift == 0:
            continue
        if shift > 0 and c != b and not zero_frame[c] and c in succ and not zero_frame[succ[c]]:
            out[i] = (b, c, succ[c])
            applied[i] = True
        elif shift < 0 and a != b and not zero_frame[a] and a in pred and not zero_frame[pred[a]]:
            out[i] = (pred[a], a, b)
            applied[i] = True
    return out, applied
