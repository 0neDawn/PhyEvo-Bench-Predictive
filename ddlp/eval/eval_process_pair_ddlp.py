"""
eval_anomaly_pair_ddlp_process.py

用途：
    与 eval_anomaly_pair_ddlp.py（状态保持实验）共用同一套指标定义和处理流程，
    仅评测数据换成 process modeling 的三类场景：
        - no_contact          （无接触正常运动，作为该类整体的可能性对照）
        - post_contact_abnormal（接触后运动异常，违反动量/能量传递规律）
        - penetration          （接触过程中违反不可穿透约束）

    与状态保持脚本的两处必要差异：
    1. 违背/交互发生的参考帧（gt_frame）在 process 数据中不一定固定为 20，
       因此原脚本中硬编码的 baseline(17-19)/post(20-21) 窗口，
       改为相对 gt_frame 的可配置偏移，行为在 gt_frame=20 时与原脚本完全一致。
    2. 新增一项内嵌的“粒子可靠性质检”：用 possible 分支（不含任何违背）
       自身在交互窗口附近的 zpos_jump 幅度，检验该场景的粒子表示是否受到
       遮挡/粒子合并影响而产生了非物理性质的跳变。此项直接作为一个新的
       summary 字段输出，不改变、不替换任何已有指标，也不需要额外脚本或
       额外一次模型推理。

    除以上两处，其余全部结构、字段名、指标定义与状态保持脚本保持一致，
    以保证两节实验方法论可比、可对照解读。

运行示例：
python eval_process_pair_ddlp.py \
  --data_root $REPO/data/process_obj3d \
  --ddlp_config $REPO/ddlp/checkpoints/diffuse-ddlp-obj3d128/ddlp_hparams.json \
  --ddlp_ckpt $REPO/ddlp/checkpoints/diffuse-ddlp-obj3d128/obj3d128_ddlp.pth \
  --cond_steps 8 \
  --context_start 12 \
  --pred_steps 6 \
  --gt_frame 20 \
  --window_pre_offset 3 \
  --window_post_offset 1 \
  --fajf_sigma_k 2.0 \
  --max_frames 26 \
  --hybrid_alpha -1.0 \
  --reliability_ratio_threshold 0.5 \
  --out_json $REPO/ddlp/eval/process_ddlp_jump_fajf_full300.json
"""

import argparse
import json
import os
import sys
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from models import ObjectDynamicsDLP


_PARTICLE_EXTRACTION_WARNED = False


def warn_particle_once(msg: str) -> None:
    global _PARTICLE_EXTRACTION_WARNED
    if not _PARTICLE_EXTRACTION_WARNED:
        print(f"[Warning] {msg}")
        _PARTICLE_EXTRACTION_WARNED = True


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
    dist_matrix = torch.cdist(pc1, pc2)
    min1, _ = dist_matrix.min(dim=1)
    min2, _ = dist_matrix.min(dim=0)
    chamfer = min1.mean() + min2.mean()
    return chamfer.item()


def _format_particle_tensor(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if x is None:
        return None
    if x.dim() == 4:
        if x.shape[0] == 1:
            x = x[0]
        else:
            x = x.reshape(-1, *x.shape[2:])
    elif x.dim() == 3:
        if x.shape[0] != seq_len and seq_len > 0 and x.shape[0] % seq_len == 0:
            x = x.reshape(-1, seq_len, *x.shape[1:])[0]
    else:
        return None
    return x


@torch.no_grad()
def extract_particle_positions(
    model: ObjectDynamicsDLP,
    x_seq: torch.Tensor,
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = x_seq.shape[1]
    errors = []

    def _parse_from_dict(out: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = None
        obj_on = None
        for k in ("z", "kp_pos", "mu", "z_base"):
            if isinstance(out, dict) and k in out:
                pos = out[k]
                break
        for k in ("obj_on", "z_obj_on", "presence", "kp_alpha"):
            if isinstance(out, dict) and k in out:
                obj_on = out[k]
                break

        pos = _format_particle_tensor(pos, seq_len)
        if pos is not None and pos.shape[-1] > 2:
            pos = pos[..., :2]
        if obj_on is not None:
            obj_on = _format_particle_tensor(obj_on, seq_len)
            if obj_on is not None and obj_on.shape[-1] == 1:
                obj_on = obj_on.squeeze(-1)
        return pos, obj_on

    if hasattr(model, "fg_sequential_opt"):
        try:
            out = model.fg_sequential_opt(x_seq, deterministic=deterministic, x_prior=x_seq, reshape=False)
            pos, obj_on = _parse_from_dict(out)
            if pos is not None:
                return pos, obj_on
        except Exception as exc:
            errors.append(f"fg_sequential_opt: {exc}")

    if hasattr(model, "encode_all"):
        try:
            out = model.encode_all(x_seq, deterministic=deterministic, x_prior=x_seq, reshape=False)
            pos, obj_on = _parse_from_dict(out)
            if pos is not None:
                return pos, obj_on
        except Exception as exc:
            errors.append(f"encode_all: {exc}")

    try:
        out = model(x_seq, deterministic=deterministic, x_prior=x_seq)
        pos, obj_on = _parse_from_dict(out)
        if pos is not None:
            return pos, obj_on
    except Exception as exc:
        errors.append(f"forward: {exc}")

    err_msg = "; ".join(errors) if errors else "unknown extraction failure"
    raise RuntimeError(err_msg)


def build_motion_mask(pos_ctx: torch.Tensor, abs_threshold: float = 0.02, keep_ratio: float = 0.5) -> torch.Tensor:
    n_particles = pos_ctx.shape[1]
    if pos_ctx.shape[0] < 2:
        return torch.ones(n_particles, dtype=torch.bool, device=pos_ctx.device)

    disp = torch.norm(pos_ctx[1:] - pos_ctx[:-1], dim=-1)
    max_disp = disp.max(dim=0).values

    k = max(1, int(n_particles * keep_ratio))
    topk_vals, topk_idx = torch.topk(max_disp, k=k, largest=True)
    quantile_thresh = topk_vals.min().item()
    threshold = max(abs_threshold, quantile_thresh)

    mask = max_disp >= threshold
    if mask.sum().item() == 0:
        mask[topk_idx] = True
    return mask


@torch.no_grad()
def anomaly_score_particle(model, video, cond_steps, device):
    t = video.shape[0]
    if t <= cond_steps:
        warn_particle_once("Particle scoring skipped: video too short for cond_steps.")
        return None

    x = video.unsqueeze(0).to(device)
    pred = model.sample(x, num_steps=t - cond_steps, cond_steps=cond_steps, deterministic=True).clamp(0, 1)

    try:
        gt_pos, gt_obj = extract_particle_positions(model, x, deterministic=True)
        pred_pos, pred_obj = extract_particle_positions(model, pred, deterministic=True)
    except Exception as exc:
        warn_particle_once(f"Particle extraction failed; skipping corresponding pair. Details: {exc}")
        return None

    if gt_pos is None or pred_pos is None:
        warn_particle_once("Particle extraction returned None; skipping corresponding pair.")
        return None

    horizon = min(gt_pos.shape[0], pred_pos.shape[0]) - cond_steps
    if horizon <= 0:
        warn_particle_once("No future horizon left after particle extraction alignment.")
        return None

    gt_ctx = gt_pos[:cond_steps]
    pred_ctx = pred_pos[:cond_steps]

    if gt_pos.shape[1] == pred_pos.shape[1]:
        shared_mask = build_motion_mask(gt_ctx)
        gt_mask = shared_mask
        pred_mask = shared_mask
    else:
        gt_mask = build_motion_mask(gt_ctx)
        pred_mask = build_motion_mask(pred_ctx)

    per_frame_chamfer = []
    for i in range(horizon):
        gt_pts = gt_pos[cond_steps + i][gt_mask]
        pred_pts = pred_pos[cond_steps + i][pred_mask]

        if gt_obj is not None and gt_obj.shape[0] > cond_steps + i and gt_obj.shape[1] == gt_pts.shape[0]:
            gt_keep = gt_obj[cond_steps + i] > 0.5
            if gt_keep.any():
                gt_pts = gt_pts[gt_keep]
        if pred_obj is not None and pred_obj.shape[0] > cond_steps + i and pred_obj.shape[1] == pred_pts.shape[0]:
            pred_keep = pred_obj[cond_steps + i] > 0.5
            if pred_keep.any():
                pred_pts = pred_pts[pred_keep]

        if gt_pts.numel() == 0 or pred_pts.numel() == 0:
            gt_pts = gt_pos[cond_steps + i]
            pred_pts = pred_pos[cond_steps + i]

        ch = chamfer_distance(gt_pts, pred_pts)
        per_frame_chamfer.append(ch)

    if not per_frame_chamfer:
        warn_particle_once("Particle Chamfer sequence is empty; skipping corresponding pair.")
        return None

    peak_local_offset = int(max(range(len(per_frame_chamfer)), key=lambda j: per_frame_chamfer[j]))
    peak_chamfer = float(per_frame_chamfer[peak_local_offset])
    peak_frame = int(cond_steps + peak_local_offset)
    return peak_chamfer, per_frame_chamfer, peak_frame


def compute_localization_summary(rows: List[Dict], gt_frame: int) -> Dict:
    peak_frames = [r["peak_frame"] for r in rows if r.get("peak_frame") is not None]
    if not peak_frames:
        return {
            "peak_hit_rate_within_1": float("nan"),
            "mean_offset": float("nan"),
            "median_offset": float("nan"),
            "num_samples": 0,
        }

    offsets = [abs(int(pf) - int(gt_frame)) for pf in peak_frames]
    hits = [1.0 if off <= 1 else 0.0 for off in offsets]
    return {
        "peak_hit_rate_within_1": float(sum(hits) / len(hits)),
        "mean_offset": float(sum(offsets) / len(offsets)),
        "median_offset": float(statistics.median(offsets)),
        "num_samples": len(offsets),
    }


def _window_mean_from_abs_frames(series: List[float], abs_start: int, abs_end: int, global_start: int) -> Optional[float]:
    vals = []
    for abs_f in range(abs_start, abs_end + 1):
        local = abs_f - global_start
        if 0 <= local < len(series):
            vals.append(float(series[local]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _window_vals_from_abs_frames(series: List[float], abs_start: int, abs_end: int, global_start: int) -> List[float]:
    vals = []
    for abs_f in range(abs_start, abs_end + 1):
        local = abs_f - global_start
        if 0 <= local < len(series):
            vals.append(float(series[local]))
    return vals


def first_anomaly_jump_frame(
    series: List[float],
    global_start: int,
    gt_frame: int,
    baseline_start: int,
    baseline_end: int,
    sigma_k: float = 2.0,
) -> Optional[int]:
    baseline_vals = _window_vals_from_abs_frames(series, baseline_start, baseline_end, global_start)
    if len(baseline_vals) == 0:
        return None
    mu = float(sum(baseline_vals) / len(baseline_vals))
    if len(baseline_vals) > 1:
        sigma = float(statistics.pstdev(baseline_vals))
    else:
        sigma = 0.0
    threshold = mu + sigma_k * sigma

    for abs_f in range(gt_frame, global_start + len(series)):
        local = abs_f - global_start
        if 0 <= local < len(series) and float(series[local]) > threshold:
            return int(abs_f)
    return None


def compute_fajf_summary(rows: List[Dict]) -> Dict:
    valid = [r for r in rows if r.get("fajf_frame") is not None]
    if not valid:
        return {
            "num_samples": 0,
            "fajf_hit_rate_within_1": float("nan"),
            "mean_offset": float("nan"),
            "median_offset": float("nan"),
        }
    offsets = [float(r["fajf_offset"]) for r in valid]
    hits = [1.0 if r.get("fajf_hit_within_1") else 0.0 for r in valid]
    return {
        "num_samples": len(valid),
        "fajf_hit_rate_within_1": float(sum(hits) / len(hits)),
        "mean_offset": float(sum(offsets) / len(offsets)),
        "median_offset": float(statistics.median(offsets)),
    }


def sorted_pngs(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]

    def key(name: str):
        stem = os.path.splitext(name)[0]
        return int(stem) if stem.isdigit() else stem

    files.sort(key=key)
    return [os.path.join(folder, f) for f in files]


def discover_pairs(root: str) -> List[Dict[str, str]]:
    # 顶层子目录名即为三类 process 数据类型：
    #   no_contact / post_contact_abnormal / penetration
    # 与状态保持脚本使用完全相同的目录发现逻辑，未做任何改动。
    pairs = []
    for anomaly_type in sorted(os.listdir(root)):
        type_dir = os.path.join(root, anomaly_type)
        if not os.path.isdir(type_dir):
            continue
        for scene_name in sorted(os.listdir(type_dir)):
            scene_dir = os.path.join(type_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            pos_dir = os.path.join(scene_dir, "possible")
            imp_dir = os.path.join(scene_dir, "impossible")
            if not (os.path.isdir(pos_dir) and os.path.isdir(imp_dir)):
                continue
            if len(sorted_pngs(pos_dir)) == 0 or len(sorted_pngs(imp_dir)) == 0:
                continue
            pairs.append(
                {
                    "type": anomaly_type,
                    "scene": scene_name,
                    "possible": pos_dir,
                    "impossible": imp_dir,
                }
            )
    return pairs


def load_video_tensor(folder: str, image_size: int, max_frames: int = -1) -> torch.Tensor:
    frame_paths = sorted_pngs(folder)
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    to_tensor = transforms.ToTensor()
    frames = []
    for fp in frame_paths:
        im = Image.open(fp).convert("RGB")
        im = im.resize((image_size, image_size))
        frames.append(to_tensor(im)[:3])
    if not frames:
        raise RuntimeError(f"No frames found in {folder}")
    return torch.stack(frames, dim=0).float()


def pairwise_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    wins = 0.0
    total = 0
    for p in pos_scores:
        for n in neg_scores:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total > 0 else float("nan")


def build_model(config: Dict, checkpoint_path: str, device: torch.device) -> ObjectDynamicsDLP:
    model = ObjectDynamicsDLP(
        cdim=config["ch"],
        enc_channels=config["enc_channels"],
        prior_channels=config["prior_channels"],
        image_size=config["image_size"],
        n_kp=config["n_kp"],
        learned_feature_dim=config["learned_feature_dim"],
        pad_mode=config["pad_mode"],
        sigma=config["sigma"],
        dropout=config["dropout"],
        patch_size=config["patch_size"],
        n_kp_enc=config["n_kp_enc"],
        n_kp_prior=config["n_kp_prior"],
        kp_range=config["kp_range"],
        kp_activation=config["kp_activation"],
        anchor_s=config["anchor_s"],
        use_resblock=config["use_resblock"],
        timestep_horizon=config["timestep_horizon"],
        predict_delta=config["predict_delta"],
        scale_std=config["scale_std"],
        offset_std=config["offset_std"],
        obj_on_alpha=config["obj_on_alpha"],
        obj_on_beta=config["obj_on_beta"],
        pint_heads=config["pint_heads"],
        pint_layers=config["pint_layers"],
        pint_dim=config["pint_dim"],
        use_correlation_heatmaps=config["use_correlation_heatmaps"],
        enable_enc_attn=config["enable_enc_attn"],
        filtering_heuristic=config["filtering_heuristic"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def anomaly_score(
    model: ObjectDynamicsDLP,
    video: torch.Tensor,
    cond_steps: int,
    event_frame: int,
    event_window: int,
    device: torch.device,
) -> Dict[str, float]:
    t = video.shape[0]
    if t <= cond_steps:
        raise ValueError(f"video too short ({t}) for cond_steps={cond_steps}")

    x = video.unsqueeze(0).to(device)
    pred, z_out = model.sample(
        x,
        num_steps=t - cond_steps,
        cond_steps=cond_steps,
        deterministic=True,
        return_z=True,
    )
    pred = pred.clamp(0, 1)

    event_start = max(cond_steps, event_frame)
    event_end = t if event_window <= 0 else min(t, event_start + event_window)

    event_abs = (pred[:, event_start:event_end] - x[:, event_start:event_end]).abs()
    event_per_frame = event_abs.mean(dim=(0, 2, 3, 4))
    err_event_mean = event_per_frame.mean().item()
    err_event_peak = event_per_frame.max().item()

    future_abs = (pred[:, cond_steps:] - x[:, cond_steps:]).abs()
    err_future_mean = future_abs.mean().item()

    gt_fg = model.fg_sequential_opt(x, deterministic=True, x_prior=x, reshape=False)
    gt_z = gt_fg["z"][:, cond_steps:]
    gt_obj = gt_fg["obj_on"][:, cond_steps:]
    pred_z = z_out["z_pos"][:, -gt_z.shape[1]:]
    pred_obj = z_out["z_obj_on"][:, -gt_obj.shape[1]:]

    horizon = min(gt_z.shape[1], pred_z.shape[1])
    gt_z = gt_z[:, :horizon]
    gt_obj = gt_obj[:, :horizon]
    pred_z = pred_z[:, :horizon]
    pred_obj = pred_obj[:, :horizon]

    z_pos_l1 = (pred_z - gt_z).abs().mean(dim=-1)
    obj_w = 0.5 * (pred_obj + gt_obj)
    weighted_pos = (z_pos_l1 * obj_w).sum(dim=-1) / (obj_w.sum(dim=-1) + 1e-8)
    obj_l1 = (pred_obj - gt_obj).abs().mean(dim=-1)

    event_start_lat = max(0, event_start - cond_steps)
    event_end_lat = horizon if event_window <= 0 else min(horizon, event_start_lat + event_window)
    if event_end_lat <= event_start_lat:
        event_start_lat = 0
        event_end_lat = horizon

    weighted_pos_event = weighted_pos[:, event_start_lat:event_end_lat]
    obj_l1_event = obj_l1[:, event_start_lat:event_end_lat]

    zpos_event_mean = weighted_pos_event.mean().item()
    zpos_event_peak = weighted_pos_event.max().item()
    obj_event_mean = obj_l1_event.mean().item()
    obj_event_peak = obj_l1_event.max().item()

    gt_fg_all = model.fg_sequential_opt(x, deterministic=True, x_prior=x, reshape=False)
    gt_z_all = gt_fg_all["z"]
    gt_obj_all = gt_fg_all["obj_on"]
    pred_z_all = z_out["z_pos"]
    pred_obj_all = z_out["z_obj_on"]
    full_h = min(gt_z_all.shape[1], pred_z_all.shape[1])
    gt_z_all = gt_z_all[:, -full_h:]
    gt_obj_all = gt_obj_all[:, -full_h:]
    pred_z_all = pred_z_all[:, -full_h:]
    pred_obj_all = pred_obj_all[:, -full_h:]

    z_pos_l1_all = (pred_z_all - gt_z_all).abs().mean(dim=-1)
    obj_w_all = 0.5 * (pred_obj_all + gt_obj_all)
    weighted_pos_all = (z_pos_l1_all * obj_w_all).sum(dim=-1) / (obj_w_all.sum(dim=-1) + 1e-8)
    obj_l1_all = (pred_obj_all - gt_obj_all).abs().mean(dim=-1)

    align_shift = t - full_h
    zpos_per_frame_full = [float(v) for v in weighted_pos_all[0].detach().cpu().tolist()]
    objon_per_frame_full = [float(v) for v in obj_l1_all[0].detach().cpu().tolist()]

    return {
        "event_l1_mean": err_event_mean,
        "event_l1_peak": err_event_peak,
        "future_l1_mean": err_future_mean,
        "zpos_event_mean": zpos_event_mean,
        "zpos_event_peak": zpos_event_peak,
        "objon_event_mean": obj_event_mean,
        "objon_event_peak": obj_event_peak,
        "zpos_per_frame_full": zpos_per_frame_full,
        "objon_per_frame_full": objon_per_frame_full,
        "align_shift": int(align_shift),
    }


def summarize(rows: List[Dict]) -> Dict:
    imp_scores = [r["impossible_event_l1"] for r in rows]
    pos_scores = [r["possible_event_l1"] for r in rows]

    pair_wins = []
    for r in rows:
        if r["impossible_event_l1"] > r["possible_event_l1"]:
            pair_wins.append(1.0)
        elif r["impossible_event_l1"] == r["possible_event_l1"]:
            pair_wins.append(0.5)
        else:
            pair_wins.append(0.0)

    overall = {
        "num_pairs": len(rows),
        "pair_accuracy": float(sum(pair_wins) / len(pair_wins)) if pair_wins else float("nan"),
        "auc_impossible_vs_possible": pairwise_auc(imp_scores, pos_scores),
        "mean_impossible_event_l1": float(sum(imp_scores) / len(imp_scores)) if imp_scores else float("nan"),
        "mean_possible_event_l1": float(sum(pos_scores) / len(pos_scores)) if pos_scores else float("nan"),
    }

    by_type_rows: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_type_rows[r["type"]].append(r)

    by_type = {}
    for tname, trows in by_type_rows.items():
        imp_t = [r["impossible_event_l1"] for r in trows]
        pos_t = [r["possible_event_l1"] for r in trows]
        wins_t = []
        for r in trows:
            if r["impossible_event_l1"] > r["possible_event_l1"]:
                wins_t.append(1.0)
            elif r["impossible_event_l1"] == r["possible_event_l1"]:
                wins_t.append(0.5)
            else:
                wins_t.append(0.0)
        by_type[tname] = {
            "num_pairs": len(trows),
            "pair_accuracy": float(sum(wins_t) / len(wins_t)) if wins_t else float("nan"),
            "auc_impossible_vs_possible": pairwise_auc(imp_t, pos_t),
            "mean_impossible_event_l1": float(sum(imp_t) / len(imp_t)) if imp_t else float("nan"),
            "mean_possible_event_l1": float(sum(pos_t) / len(pos_t)) if pos_t else float("nan"),
        }

    return {"overall": overall, "by_type": by_type}


def summarize_from_scores(rows: List[Dict], pos_key: str, imp_key: str) -> Dict:
    imp_scores = [r[imp_key] for r in rows]
    pos_scores = [r[pos_key] for r in rows]

    pair_wins = []
    for r in rows:
        if r[imp_key] > r[pos_key]:
            pair_wins.append(1.0)
        elif r[imp_key] == r[pos_key]:
            pair_wins.append(0.5)
        else:
            pair_wins.append(0.0)

    overall = {
        "num_pairs": len(rows),
        "pair_accuracy": float(sum(pair_wins) / len(pair_wins)) if pair_wins else float("nan"),
        "auc_impossible_vs_possible": pairwise_auc(imp_scores, pos_scores),
        f"mean_{imp_key}": float(sum(imp_scores) / len(imp_scores)) if imp_scores else float("nan"),
        f"mean_{pos_key}": float(sum(pos_scores) / len(pos_scores)) if pos_scores else float("nan"),
    }

    by_type_rows: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_type_rows[r["type"]].append(r)

    by_type = {}
    for tname, trows in by_type_rows.items():
        imp_t = [r[imp_key] for r in trows]
        pos_t = [r[pos_key] for r in trows]
        wins_t = []
        for r in trows:
            if r[imp_key] > r[pos_key]:
                wins_t.append(1.0)
            elif r[imp_key] == r[pos_key]:
                wins_t.append(0.5)
            else:
                wins_t.append(0.0)
        by_type[tname] = {
            "num_pairs": len(trows),
            "pair_accuracy": float(sum(wins_t) / len(wins_t)) if wins_t else float("nan"),
            "auc_impossible_vs_possible": pairwise_auc(imp_t, pos_t),
            f"mean_{imp_key}": float(sum(imp_t) / len(imp_t)) if imp_t else float("nan"),
            f"mean_{pos_key}": float(sum(pos_t) / len(pos_t)) if pos_t else float("nan"),
        }

    return {"overall": overall, "by_type": by_type}


def add_two_sided_scores(rows: List[Dict]) -> None:
    pos_global = [r["possible_event_l1"] for r in rows]
    global_center = statistics.median(pos_global) if pos_global else 0.0

    type_centers = {}
    by_type_pos: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_type_pos[r["type"]].append(r["possible_event_l1"])
    for tname, vals in by_type_pos.items():
        type_centers[tname] = statistics.median(vals)

    for r in rows:
        c_global = global_center
        c_type = type_centers[r["type"]]
        r["possible_event_l1_absdev_global"] = abs(r["possible_event_l1"] - c_global)
        r["impossible_event_l1_absdev_global"] = abs(r["impossible_event_l1"] - c_global)
        r["possible_event_l1_absdev_type"] = abs(r["possible_event_l1"] - c_type)
        r["impossible_event_l1_absdev_type"] = abs(r["impossible_event_l1"] - c_type)


def add_hybrid_scores(rows: List[Dict], alpha: float) -> None:
    for r in rows:
        r["possible_event_hybrid_type"] = r["possible_event_l1_absdev_type"] + alpha * r["possible_event_l1"]
        r["impossible_event_hybrid_type"] = r["impossible_event_l1_absdev_type"] + alpha * r["impossible_event_l1"]


def compute_particle_reliability_summary(rows: List[Dict], ratio_threshold: float) -> Dict:
    """
    与状态保持实验共用同一套指标（possible_zpos_jump / impossible_zpos_jump），
    仅作为一次事后统计，不引入新指标、不需要额外一次模型推理：

    possible 分支不含任何物理违背，其 zpos_jump 理论上应接近噪声水平。
    若 possible 分支中 |zpos_jump| 达到 impossible 分支 |zpos_jump| 中位数的
    ratio_threshold 倍以上，说明该场景的粒子表示很可能在接触/遮挡窗口附近
    发生了粒子合并或轨迹跳变，与真实物理违背无关，需要在解读 zpos 类指标时
    加以说明。该统计直接嵌入本脚本输出，与其余 summary 字段并列，不单独
    产出文件，保持与状态保持实验一致的产出结构。
    """
    jump_rows = [
        r for r in rows
        if r.get("possible_zpos_jump") is not None and r.get("impossible_zpos_jump") is not None
    ]
    if not jump_rows:
        return {
            "num_scenes": 0,
            "impossible_abs_median": float("nan"),
            "threshold_used": float("nan"),
            "noisy_ratio": float("nan"),
            "by_type": {},
        }

    def _stats(sub_rows: List[Dict]) -> Dict:
        imp_abs = [abs(r["impossible_zpos_jump"]) for r in sub_rows]
        pos_abs = [abs(r["possible_zpos_jump"]) for r in sub_rows]
        imp_median = statistics.median(imp_abs)
        threshold = ratio_threshold * imp_median
        noisy = [1.0 if v > threshold else 0.0 for v in pos_abs]
        return {
            "num_scenes": len(sub_rows),
            "impossible_abs_median": float(imp_median),
            "threshold_used": float(threshold),
            "noisy_ratio": float(sum(noisy) / len(noisy)) if noisy else float("nan"),
        }

    overall = _stats(jump_rows)

    by_type_rows: Dict[str, List[Dict]] = defaultdict(list)
    for r in jump_rows:
        by_type_rows[r["type"]].append(r)
    by_type = {tname: _stats(trows) for tname, trows in by_type_rows.items()}

    overall["by_type"] = by_type
    return overall


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DDLP on process-modeling anomaly pairs (no_contact / post_contact_abnormal / penetration)"
    )
    parser.add_argument("--data_root", type=str, required=True, help="root of process_3d dataset")
    parser.add_argument("--ddlp_config", type=str, required=True, help="path to ddlp hparams json")
    parser.add_argument("--ddlp_ckpt", type=str, required=True, help="path to ddlp .pth checkpoint")
    parser.add_argument("--cond_steps", type=int, default=8, help="number of context frames")
    parser.add_argument(
        "--context_start", type=int, default=-1,
        help="absolute start frame of context window; if >=0, evaluate [context_start : context_start+cond_steps+pred_steps)",
    )
    parser.add_argument("--pred_steps", type=int, default=6, help="number of predicted frames for context window mode")
    parser.add_argument("--event_frame", type=int, default=8, help="score starts from this frame index (0-based)")
    parser.add_argument("--event_window", type=int, default=6, help="frames to score from event_frame; <=0 means all future")
    parser.add_argument(
        "--gt_frame", type=int, default=20,
        help="ground-truth contact/interaction frame index for localization (与状态保持脚本同义参数)",
    )
    parser.add_argument(
        "--window_pre_offset", type=int, default=3,
        help="baseline窗口为 [gt_frame-window_pre_offset, gt_frame-1]，默认3对应状态保持脚本中硬编码的17-19（gt_frame=20时）",
    )
    parser.add_argument(
        "--window_post_offset", type=int, default=1,
        help="post窗口为 [gt_frame, gt_frame+window_post_offset]，默认1对应状态保持脚本中硬编码的20-21（gt_frame=20时）",
    )
    parser.add_argument("--fajf_sigma_k", type=float, default=2.0, help="sigma multiplier for FAJF threshold mu + k*sigma")
    parser.add_argument("--max_frames", type=int, default=14, help="clip to first N frames; <=0 means all")
    parser.add_argument("--limit_pairs", type=int, default=-1, help="limit number of scene pairs for quick test")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument(
        "--hybrid_alpha", type=float, default=-1.05,
        help="alpha for hybrid score: absdev_type + alpha * raw_event_l1",
    )
    parser.add_argument(
        "--reliability_ratio_threshold", type=float, default=0.5,
        help="粒子可靠性质检阈值：possible分支|zpos_jump|超过 ratio_threshold × impossible分支|zpos_jump|中位数 视为可疑",
    )
    parser.add_argument("--out_json", type=str, default="./eval/process_ddlp_results.json", help="output json path")
    args = parser.parse_args()

    with open(args.ddlp_config, "r") as f:
        config = json.load(f)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(config, args.ddlp_ckpt, device)
    print(f"Loaded DDLP from {args.ddlp_ckpt}")

    pairs = discover_pairs(args.data_root)
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]
    if not pairs:
        raise RuntimeError(f"No possible/impossible pairs found under {args.data_root}")

    print(f"Found {len(pairs)} scene pairs")

    # baseline/post窗口相对gt_frame计算，替代原脚本硬编码的17-19/20-21，
    # gt_frame=20、window_pre_offset=3、window_post_offset=1时行为完全一致。
    baseline_start = args.gt_frame - args.window_pre_offset
    baseline_end = args.gt_frame - 1
    post_start = args.gt_frame
    post_end = args.gt_frame + args.window_post_offset

    rows = []
    particle_skipped = 0
    for i, pair in enumerate(pairs, start=1):
        possible = load_video_tensor(pair["possible"], image_size=config["image_size"], max_frames=args.max_frames)
        impossible = load_video_tensor(pair["impossible"], image_size=config["image_size"], max_frames=args.max_frames)

        t = min(possible.shape[0], impossible.shape[0])
        possible = possible[:t]
        impossible = impossible[:t]

        if t <= args.cond_steps:
            print(f"[skip] {pair['type']}/{pair['scene']} too short: {t} frames")
            continue

        eval_event_frame = args.event_frame
        eval_event_window = args.event_window

        if args.context_start >= 0:
            required_t = args.context_start + args.cond_steps + args.pred_steps
            if t < required_t:
                print(
                    f"[skip] {pair['type']}/{pair['scene']} too short for context window: "
                    f"need {required_t}, got {t}"
                )
                continue
            possible = possible[args.context_start:required_t]
            impossible = impossible[args.context_start:required_t]
            eval_event_frame = args.cond_steps
            eval_event_window = args.pred_steps

        pos_metrics = anomaly_score(
            model, possible, args.cond_steps, eval_event_frame, eval_event_window, device
        )
        imp_metrics = anomaly_score(
            model, impossible, args.cond_steps, eval_event_frame, eval_event_window, device
        )

        global_start = args.context_start if args.context_start >= 0 else 0

        pos_zpos_series = pos_metrics["zpos_per_frame_full"]
        imp_zpos_series = imp_metrics["zpos_per_frame_full"]
        pos_obj_series = pos_metrics["objon_per_frame_full"]
        imp_obj_series = imp_metrics["objon_per_frame_full"]

        pos_pre = _window_mean_from_abs_frames(pos_zpos_series, baseline_start, baseline_end, global_start)
        pos_post = _window_mean_from_abs_frames(pos_zpos_series, post_start, post_end, global_start)
        imp_pre = _window_mean_from_abs_frames(imp_zpos_series, baseline_start, baseline_end, global_start)
        imp_post = _window_mean_from_abs_frames(imp_zpos_series, post_start, post_end, global_start)

        pos_jump = None if (pos_pre is None or pos_post is None) else float(pos_post - pos_pre)
        imp_jump = None if (imp_pre is None or imp_post is None) else float(imp_post - imp_pre)

        pos_jump_norm = None if pos_jump is None else float(pos_jump / (abs(pos_pre) + 1e-3))
        imp_jump_norm = None if imp_jump is None else float(imp_jump / (abs(imp_pre) + 1e-3))

        pos_obj_post = _window_mean_from_abs_frames(pos_obj_series, post_start, post_end, global_start)
        imp_obj_post = _window_mean_from_abs_frames(imp_obj_series, post_start, post_end, global_start)

        pos_joint = None if (pos_jump is None or pos_obj_post is None) else float(pos_jump * (1.0 - pos_obj_post))
        imp_joint = None if (imp_jump is None or imp_obj_post is None) else float(imp_jump * (1.0 - imp_obj_post))

        fajf_frame = first_anomaly_jump_frame(
            imp_zpos_series,
            global_start=global_start,
            gt_frame=args.gt_frame,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            sigma_k=args.fajf_sigma_k,
        )
        fajf_offset = None if fajf_frame is None else int(abs(int(fajf_frame) - int(args.gt_frame)))
        fajf_hit = None if fajf_frame is None else bool(fajf_offset <= 1)

        pos_particle = anomaly_score_particle(model, possible, args.cond_steps, device)
        imp_particle = anomaly_score_particle(model, impossible, args.cond_steps, device)

        if pos_particle is None or imp_particle is None:
            particle_skipped += 1
            print(f"[Warning] skipping particle branch for {pair['type']}/{pair['scene']} due to particle extraction failure")

        row = {
            "type": pair["type"],
            "scene": pair["scene"],
            "num_frames": int(t),
            "possible_event_l1": pos_metrics["event_l1_mean"],
            "impossible_event_l1": imp_metrics["event_l1_mean"],
            "possible_event_l1_peak": pos_metrics["event_l1_peak"],
            "impossible_event_l1_peak": imp_metrics["event_l1_peak"],
            "possible_future_l1": pos_metrics["future_l1_mean"],
            "impossible_future_l1": imp_metrics["future_l1_mean"],
            "possible_zpos_event_mean": pos_metrics["zpos_event_mean"],
            "impossible_zpos_event_mean": imp_metrics["zpos_event_mean"],
            "possible_zpos_event_peak": pos_metrics["zpos_event_peak"],
            "impossible_zpos_event_peak": imp_metrics["zpos_event_peak"],
            "possible_objon_event_mean": pos_metrics["objon_event_mean"],
            "impossible_objon_event_mean": imp_metrics["objon_event_mean"],
            "possible_objon_event_peak": pos_metrics["objon_event_peak"],
            "impossible_objon_event_peak": imp_metrics["objon_event_peak"],
            "possible_peak_chamfer": None if pos_particle is None else float(pos_particle[0]),
            "impossible_peak_chamfer": None if imp_particle is None else float(imp_particle[0]),
            "possible_per_frame_chamfer": None if pos_particle is None else [float(v) for v in pos_particle[1]],
            "impossible_per_frame_chamfer": None if imp_particle is None else [float(v) for v in imp_particle[1]],
            "peak_frame": None if imp_particle is None else int(imp_particle[2]),
            "peak_offset": None if imp_particle is None else int(abs(int(imp_particle[2]) - int(args.gt_frame))),
            "peak_hit_within_1": None if imp_particle is None else bool(abs(int(imp_particle[2]) - int(args.gt_frame)) <= 1),
            "possible_zpos_jump": pos_jump,
            "impossible_zpos_jump": imp_jump,
            "possible_zpos_jump_norm": pos_jump_norm,
            "impossible_zpos_jump_norm": imp_jump_norm,
            "possible_objon_post_mean": pos_obj_post,
            "impossible_objon_post_mean": imp_obj_post,
            "possible_joint_jump_objon": pos_joint,
            "impossible_joint_jump_objon": imp_joint,
            "fajf_frame": fajf_frame,
            "fajf_offset": fajf_offset,
            "fajf_hit_within_1": fajf_hit,
            "margin_impossible_minus_possible": imp_metrics["event_l1_mean"] - pos_metrics["event_l1_mean"],
            "pair_hit": bool(imp_metrics["event_l1_mean"] > pos_metrics["event_l1_mean"]),
        }
        rows.append(row)

        if i % 20 == 0 or i == len(pairs):
            print(f"Processed {i}/{len(pairs)} pairs")

    if not rows:
        raise RuntimeError("No valid pairs to evaluate")

    summary = summarize(rows)
    add_two_sided_scores(rows)
    add_hybrid_scores(rows, alpha=args.hybrid_alpha)
    summary_absdev_global = summarize_from_scores(
        rows, "possible_event_l1_absdev_global", "impossible_event_l1_absdev_global"
    )
    summary_absdev_type = summarize_from_scores(
        rows, "possible_event_l1_absdev_type", "impossible_event_l1_absdev_type"
    )
    summary_hybrid_type = summarize_from_scores(
        rows, "possible_event_hybrid_type", "impossible_event_hybrid_type"
    )
    summary_peak_l1 = summarize_from_scores(
        rows, "possible_event_l1_peak", "impossible_event_l1_peak"
    )
    summary_latent_zpos_mean = summarize_from_scores(
        rows, "possible_zpos_event_mean", "impossible_zpos_event_mean"
    )
    summary_latent_zpos_peak = summarize_from_scores(
        rows, "possible_zpos_event_peak", "impossible_zpos_event_peak"
    )
    summary_latent_objon_mean = summarize_from_scores(
        rows, "possible_objon_event_mean", "impossible_objon_event_mean"
    )
    summary_latent_objon_peak = summarize_from_scores(
        rows, "possible_objon_event_peak", "impossible_objon_event_peak"
    )

    jump_rows = [
        r for r in rows
        if r.get("possible_zpos_jump") is not None and r.get("impossible_zpos_jump") is not None
    ]
    if jump_rows:
        summary_zpos_jump = summarize_from_scores(jump_rows, "possible_zpos_jump", "impossible_zpos_jump")
        summary_zpos_jump_norm = summarize_from_scores(
            jump_rows, "possible_zpos_jump_norm", "impossible_zpos_jump_norm"
        )
        summary_joint_jump_objon = summarize_from_scores(
            jump_rows, "possible_joint_jump_objon", "impossible_joint_jump_objon"
        )
        summary_fajf = compute_fajf_summary(jump_rows)
    else:
        empty_summary = {"overall": {"num_pairs": 0, "pair_accuracy": float("nan"), "auc_impossible_vs_possible": float("nan")}, "by_type": {}}
        summary_zpos_jump = empty_summary
        summary_zpos_jump_norm = empty_summary
        summary_joint_jump_objon = empty_summary
        summary_fajf = {"num_samples": 0, "fajf_hit_rate_within_1": float("nan"), "mean_offset": float("nan"), "median_offset": float("nan")}

    particle_rows = [
        r for r in rows
        if r.get("possible_peak_chamfer") is not None and r.get("impossible_peak_chamfer") is not None
    ]
    if particle_rows:
        particle_cls = summarize_from_scores(particle_rows, "possible_peak_chamfer", "impossible_peak_chamfer")
        particle_loc = compute_localization_summary(particle_rows, args.gt_frame)
    else:
        particle_cls = {"overall": {"num_pairs": 0, "pair_accuracy": float("nan"), "auc_impossible_vs_possible": float("nan")}, "by_type": {}}
        particle_loc = {"peak_hit_rate_within_1": float("nan"), "mean_offset": float("nan"), "median_offset": float("nan"), "num_samples": 0}

    summary_particle_peak = {"classification": particle_cls, "localization": particle_loc}

    # 新增：粒子可靠性质检，直接复用 possible/impossible_zpos_jump 字段，
    # 不新增模型推理，不新增数据，仅作为一次内嵌的事后统计并入输出。
    summary_particle_reliability = compute_particle_reliability_summary(
        rows, ratio_threshold=args.reliability_ratio_threshold
    )

    out = {
        "settings": {
            "data_root": args.data_root,
            "ddlp_config": args.ddlp_config,
            "ddlp_ckpt": args.ddlp_ckpt,
            "cond_steps": args.cond_steps,
            "context_start": args.context_start,
            "pred_steps": args.pred_steps,
            "event_frame": args.event_frame,
            "event_window": args.event_window,
            "gt_frame": args.gt_frame,
            "window_pre_offset": args.window_pre_offset,
            "window_post_offset": args.window_post_offset,
            "max_frames": args.max_frames,
            "hybrid_alpha": args.hybrid_alpha,
            "reliability_ratio_threshold": args.reliability_ratio_threshold,
            "device": str(device),
            "num_pairs_used": len(rows),
            "particle_pairs_used": len(particle_rows),
            "particle_pairs_skipped": particle_skipped,
        },
        "summary": summary,
        "summary_absdev_global": summary_absdev_global,
        "summary_absdev_type": summary_absdev_type,
        "summary_hybrid_type": summary_hybrid_type,
        "summary_peak_l1": summary_peak_l1,
        "summary_latent_zpos_mean": summary_latent_zpos_mean,
        "summary_latent_zpos_peak": summary_latent_zpos_peak,
        "summary_latent_objon_mean": summary_latent_objon_mean,
        "summary_latent_objon_peak": summary_latent_objon_peak,
        "summary_particle_peak": summary_particle_peak,
        "summary_zpos_jump": summary_zpos_jump,
        "summary_zpos_jump_norm": summary_zpos_jump_norm,
        "summary_joint_jump_objon": summary_joint_jump_objon,
        "summary_fajf": summary_fajf,
        "summary_particle_reliability": summary_particle_reliability,
        "per_pair": rows,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("==== Summary ====")
    print(json.dumps(summary["overall"], indent=2))
    print("==== Summary (Two-sided absdev, type-centered) ====")
    print(json.dumps(summary_absdev_type["overall"], indent=2))
    print("==== Summary (Hybrid, type-centered) ====")
    print(json.dumps(summary_hybrid_type["overall"], indent=2))
    print("==== Summary (Peak L1) ====")
    print(json.dumps(summary_peak_l1["overall"], indent=2))
    print("==== Summary (Latent zpos mean / peak) ====")
    print(json.dumps(summary_latent_zpos_mean["overall"], indent=2))
    print(json.dumps(summary_latent_zpos_peak["overall"], indent=2))
    print("==== Summary (Latent obj_on mean / peak) ====")
    print(json.dumps(summary_latent_objon_mean["overall"], indent=2))
    print(json.dumps(summary_latent_objon_peak["overall"], indent=2))
    print("==== Summary (Particle peak Chamfer + localization) ====")
    print(json.dumps(summary_particle_peak["classification"]["overall"], indent=2))
    print(json.dumps(summary_particle_peak["localization"], indent=2))
    print("==== Summary (Jump / Joint / FAJF) ====")
    print(json.dumps(summary_zpos_jump["overall"], indent=2))
    print(json.dumps(summary_zpos_jump_norm["overall"], indent=2))
    print(json.dumps(summary_joint_jump_objon["overall"], indent=2))
    print(json.dumps(summary_fajf, indent=2))
    print("==== Summary (Particle reliability check, possible-branch self-test) ====")
    print(json.dumps(
        {k: v for k, v in summary_particle_reliability.items() if k != "by_type"}, indent=2
    ))
    print(json.dumps(summary_particle_reliability.get("by_type", {}), indent=2))
    print(f"Saved full results to: {args.out_json}")


if __name__ == "__main__":
    main()