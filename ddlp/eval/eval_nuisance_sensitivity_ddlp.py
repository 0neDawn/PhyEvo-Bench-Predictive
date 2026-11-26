"""
Evaluate DDLP encoder zpos stability under nuisance appearance perturbations.

Purpose:
- Test whether DDLP object-position encoding is stable under non-physical visual changes.
- This does NOT evaluate prediction error.
- It directly compares DDLP-encoded particle positions between clean and perturbed frames.

Protocol:
- Load possible videos only.
- Keep context conceptually irrelevant; this is encoder-level stability.
- Perturb future frames only for consistency with previous protocol.
- Encode clean and perturbed videos.
- Compare zpos using Chamfer distance to avoid particle-order mismatch.

Perturbations:
1) brightness_future
2) contrast_future
3) gaussian_noise_future
4) blur_future
5) bg_texture_future
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from eval.eval_anomaly_pair_ddlp import (  # noqa: E402
    build_model,
    discover_pairs,
    load_video_tensor,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def estimate_background_mask(video: torch.Tensor, threshold: float = 0.18) -> torch.Tensor:
    """
    video: [T, C, H, W]
    return: [T, 1, H, W], 1 = confident background
    """
    t, c, h, w = video.shape
    k = max(4, h // 16)

    corners = torch.cat(
        [
            video[:, :, :k, :k].reshape(t, c, -1),
            video[:, :, :k, -k:].reshape(t, c, -1),
            video[:, :, -k:, :k].reshape(t, c, -1),
            video[:, :, -k:, -k:].reshape(t, c, -1),
        ],
        dim=-1,
    )

    bg_color = corners.median(dim=-1).values.view(t, c, 1, 1)
    dist = (video - bg_color).abs().mean(dim=1, keepdim=True)

    mask = (dist < threshold).float()

    # Conservative erosion: only keep confident background
    inv = 1.0 - mask
    inv_dilated = F.max_pool2d(inv, kernel_size=7, stride=1, padding=3)
    mask_eroded = 1.0 - inv_dilated

    return mask_eroded.clamp(0.0, 1.0)


def perturb_brightness_future(video: torch.Tensor, cond_steps: int, delta: float = 0.06) -> torch.Tensor:
    out = video.clone()
    out[cond_steps:] = clamp01(out[cond_steps:] + delta)
    return out


def perturb_contrast_future(video: torch.Tensor, cond_steps: int, factor: float = 1.08) -> torch.Tensor:
    out = video.clone()
    future = out[cond_steps:]
    mean = future.mean(dim=(2, 3), keepdim=True)
    out[cond_steps:] = clamp01((future - mean) * factor + mean)
    return out


def perturb_gaussian_noise_future(video: torch.Tensor, cond_steps: int, sigma: float = 0.015) -> torch.Tensor:
    out = video.clone()
    noise = torch.randn_like(out[cond_steps:]) * sigma
    out[cond_steps:] = clamp01(out[cond_steps:] + noise)
    return out


def perturb_blur_future(video: torch.Tensor, cond_steps: int, kernel_size: int = 3) -> torch.Tensor:
    out = video.clone()
    future = out[cond_steps:]
    blurred = F.avg_pool2d(
        future,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    out[cond_steps:] = blurred
    return out


def perturb_bg_texture_future(
    video: torch.Tensor,
    cond_steps: int,
    amp: float = 0.025,
    fx: int = 7,
    fy: int = 5,
    bg_threshold: float = 0.18,
) -> torch.Tensor:
    out = video.clone()
    future = out[cond_steps:]

    t, c, h, w = future.shape
    bg_mask = estimate_background_mask(future, threshold=bg_threshold)

    yy = torch.linspace(0.0, 2.0 * np.pi, h, dtype=video.dtype, device=video.device)
    xx = torch.linspace(0.0, 2.0 * np.pi, w, dtype=video.dtype, device=video.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")

    pattern = torch.sin(fx * gx) + torch.cos(fy * gy)
    pattern = pattern / (pattern.abs().max() + 1e-8)
    pattern = pattern.view(1, 1, h, w).repeat(t, c, 1, 1)

    out[cond_steps:] = clamp01(future + amp * pattern * bg_mask)
    return out


def recursive_find_zpos(x: Any) -> Optional[torch.Tensor]:
    """
    Try to find z position tensor from nested DDLP outputs.
    Expected shape usually: [..., N, 2] or [..., N, D] where first 2 dims are xy.
    """
    if isinstance(x, dict):
        preferred_keys = [
            "z",
            "z_pos",
            "zpos",
            "z_positions",
            "kp",
            "keypoints",
            "particles",
            "particle_positions",
            "mu",
        ]

        for k in preferred_keys:
            if k in x and torch.is_tensor(x[k]):
                t = x[k]
                if t.ndim >= 3 and t.shape[-1] >= 2:
                    return t[..., :2]

        for v in x.values():
            found = recursive_find_zpos(v)
            if found is not None:
                return found

    elif isinstance(x, (list, tuple)):
        for v in x:
            found = recursive_find_zpos(v)
            if found is not None:
                return found

    return None


@torch.no_grad()
def encode_zpos(model: torch.nn.Module, video: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    video: [T, C, H, W]
    return: [T, N, 2] on CPU

    This function is intentionally defensive because DDLP repos may use different output names.
    """
    model.eval()

    x = video.unsqueeze(0).to(device)  # [1, T, C, H, W]

    # Try common forward patterns
    try:
        out = model(x)
    except TypeError:
        try:
            out = model(x, deterministic=True)
        except TypeError:
            out = model.forward(x)

    zpos = recursive_find_zpos(out)

    if zpos is None:
        raise RuntimeError(
            "Could not find z position in model output. "
            "Print model output keys and update recursive_find_zpos()."
        )

    # Remove batch dimension if present
    if zpos.ndim == 4 and zpos.shape[0] == 1:
        zpos = zpos[0]

    # If model returns [B, something, T, N, 2], this needs manual adjustment.
    if zpos.ndim != 3 or zpos.shape[-1] != 2:
        raise RuntimeError(f"Unexpected zpos shape after extraction: {tuple(zpos.shape)}")

    return zpos.detach().cpu()


def chamfer_zpos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b: [T, N, 2]
    return: [T], Chamfer distance per frame

    Handles particle permutation mismatch.
    """
    d = torch.cdist(a, b)  # [T, N, N]
    return 0.5 * (d.min(dim=-1).values.mean(dim=-1) + d.min(dim=-2).values.mean(dim=-1))


def summarize(values: List[float]) -> Dict:
    if not values:
        return {
            "num_samples": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "num_samples": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DDLP encoder zpos nuisance stability")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ddlp_config", type=str, required=True)
    parser.add_argument("--ddlp_ckpt", type=str, required=True)

    parser.add_argument("--cond_steps", type=int, default=8)
    parser.add_argument("--context_start", type=int, default=12)
    parser.add_argument("--pred_steps", type=int, default=6)
    parser.add_argument("--max_frames", type=int, default=26)

    parser.add_argument("--limit_pairs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument(
        "--out_json",
        type=str,
        default="./eval/encoder_nuisance_zpos_ddlp.json",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    with open(args.ddlp_config, "r") as f:
        config = json.load(f)

    device = torch.device(
        "cpu" if args.cpu else "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(config, args.ddlp_ckpt, device)

    pairs = discover_pairs(args.data_root)
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]

    perturb_fns = {
        "brightness_future": lambda v: perturb_brightness_future(v, args.cond_steps),
        "contrast_future": lambda v: perturb_contrast_future(v, args.cond_steps),
        "gaussian_noise_future": lambda v: perturb_gaussian_noise_future(v, args.cond_steps),
        "blur_future": lambda v: perturb_blur_future(v, args.cond_steps),
        "bg_texture_future": lambda v: perturb_bg_texture_future(v, args.cond_steps),
    }

    scores_by_type = defaultdict(list)
    rows_by_type = defaultdict(list)

    for i, pair in enumerate(pairs, start=1):
        video = load_video_tensor(
            pair["possible"],
            image_size=config["image_size"],
            max_frames=args.max_frames,
        )

        if args.context_start >= 0:
            need = args.context_start + args.cond_steps + args.pred_steps
            if video.shape[0] < need:
                continue
            video = video[args.context_start:need]
            future_slice = slice(args.cond_steps, args.cond_steps + args.pred_steps)
            global_start = args.context_start
        else:
            if video.shape[0] < args.cond_steps + args.pred_steps:
                continue
            video = video[: args.cond_steps + args.pred_steps]
            future_slice = slice(args.cond_steps, args.cond_steps + args.pred_steps)
            global_start = 0

        try:
            z_clean = encode_zpos(model, video, device)
        except Exception as e:
            print(f"[ERROR] Failed on clean video {pair.get('possible', '')}: {e}")
            raise

        z_clean_future = z_clean[future_slice]

        for name, fn in perturb_fns.items():
            pert_video = fn(video.clone())

            try:
                z_pert = encode_zpos(model, pert_video, device)
            except Exception as e:
                print(f"[ERROR] Failed on perturb={name}, video={pair.get('possible', '')}: {e}")
                raise

            z_pert_future = z_pert[future_slice]

            dist_t = chamfer_zpos(z_clean_future, z_pert_future)
            score_mean = float(dist_t.mean().item())
            score_median = float(dist_t.median().item())
            score_max = float(dist_t.max().item())

            scores_by_type[name].append(score_mean)

            rows_by_type[name].append(
                {
                    "type": pair.get("type", ""),
                    "scene": pair.get("scene", ""),
                    "global_start": int(global_start),
                    "score_mean": score_mean,
                    "score_median": score_median,
                    "score_max": score_max,
                    "per_frame": [float(x) for x in dist_t.tolist()],
                }
            )

        if i % 20 == 0 or i == len(pairs):
            print(f"Processed {i}/{len(pairs)} possible videos")

    summary = {name: summarize(vals) for name, vals in scores_by_type.items()}

    out = {
        "settings": {
            "data_root": args.data_root,
            "ddlp_config": args.ddlp_config,
            "ddlp_ckpt": args.ddlp_ckpt,
            "cond_steps": args.cond_steps,
            "context_start": args.context_start,
            "pred_steps": args.pred_steps,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "device": str(device),
        },
        "summary": summary,
        "per_perturbation": rows_by_type,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("==== DDLP Encoder Zpos Nuisance Stability ====")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {args.out_json}")


if __name__ == "__main__":
    main()