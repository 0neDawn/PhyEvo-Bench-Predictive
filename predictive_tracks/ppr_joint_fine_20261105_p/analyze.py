"""Read measured errors to propose eta/alpha; never publish surrogate scores."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata
from . import common as c


def source_rows(model):
    values = {}
    for x in c.source()["all_configurations"][model]:
        p = x["parameters"]
        if not 0 < p["eta"] <= 1 or not 1 <= p.get("alpha", 1.) <= 4:
            continue
        if model in c.OBJECTS and not any(c.quality_key(p) == c.quality_key(c.quality(model, t)) for t in ("sp", "ie")):
            continue
        q = {k: p.get(k) for k in ("q", "presence", "mass", "min_history")} if model in c.OBJECTS else None
        # Later O records use the frozen current arithmetic. Confirmation rows
        # can be curve knots, while remaining ineligible as new discoveries.
        values[c.semantic_key(model, x["mode"], p["eta"], p.get("alpha", 1.), q)] = x
    return list(values.values())


def load_model(model):
    root = c.HERE / "analysis" / model
    if (root / "INDEX.json").exists():
        index = c.read(root / "INDEX.json")
        assert c.digest(root / "errors.npz") == index["errors_sha256"]
        with np.load(root / "errors.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return index["settings"], arrays
    from predictive_tracks.ppr_eta_radius_20261104_o.path_runtime_001 import install
    report = install()
    settings = source_rows(model)
    assert settings
    grouped, ordinary = defaultdict(list), []
    for k, x in enumerate(settings):
        if x.get("compact_case_source"):
            grouped[x["compact_case_source"]].append(k)
        else:
            ordinary.append(k)
    losses = native = refs = spans = None

    def put(k, rows):
        nonlocal losses, native, refs, spans
        rows = sorted(rows, key=lambda x: (x["benchmark"], x["archive_index"]))
        assert len(rows) == 600
        assert [(x["benchmark"], x["archive_index"]) for x in rows] == [(t, n) for t in ("ie", "sp") for n in range(300)]
        val = np.asarray([[x["possible_errors"], x["impossible_errors"]] for x in rows], np.float64)
        base = np.asarray([[x["matched_identity_possible_errors"], x["matched_identity_impossible_errors"]] for x in rows], np.float64)
        rr = np.asarray([int(x["t_ref"]) for x in rows])
        ss = np.asarray([rows[0]["frame_spans"], rows[300]["frame_spans"]])
        for start in (0, 300):
            assert all(x["frame_spans"] == rows[start]["frame_spans"] for x in rows[start:start+300])
        if losses is None:
            losses = np.empty((len(settings), *val.shape), np.float64)
            native, refs, spans = base, rr, ss
        assert val.shape == native.shape and np.array_equal(base, native) and np.array_equal(rr, refs) and np.array_equal(ss, spans)
        assert np.isfinite(val).all() and (val >= 0).all()
        losses[k] = val

    for k in ordinary:
        c.cpu_guard()
        put(k, report.records(settings[k]))
        c.status(c.HERE / "analysis_progress.json", dict(at=time.time(), stage="read_native_scores", model=model, completed=k+1, total=len(settings)))
    # One JSON read per compact case, independent of number of configurations.
    for folder, indices in grouped.items():
        c.cpu_guard()
        records = {k: [] for k in indices}
        paths = sorted(Path(folder).glob("*.json"))
        assert len(paths) == 600
        for path in paths:
            payload = c.read(path)
            for k in indices:
                row = dict(payload["native"])
                row.update(payload["settings"][settings[k]["config_id"]])
                records[k].append(row)
        for k in indices:
            put(k, records[k])
        del records
    root.mkdir(parents=True, exist_ok=True)
    temp = root / f"errors.atomic.{os.getpid()}.npz"
    with temp.open("xb") as stream:
        np.savez_compressed(stream, losses=losses, native=native, refs=refs, spans=spans)
    dest = root / "errors.npz"
    os.link(temp, dest)
    temp.unlink()
    c.save(root / "INDEX.json", dict(at=time.time(), model=model, settings=settings, errors_sha256=c.digest(dest), source_report_sha256=c.digest(c.SOURCE),
                                     scope="Measured old error arrays, used only for proposals; no new benchmark scores."))
    return settings, dict(losses=losses, native=native, refs=refs, spans=spans)


def rank_actual(model, task):
    o = c.source()["selected"][task][model]["metrics"][task]["corrected"]["overall"]
    tol = "2" if model in ("vjepa2_vitg", "vjepa2_vitl", "videomaev2") else "1"
    return (round(o["joint_hit"][tol]*300), round(o["paired_accuracy"]*300), round(o["auc"]*180000), -(o["localization_mae"] if o["localization_mae"] is not None else 1e30))


def evaluate(values, task, arrays, model):
    from predictive_tracks.ppr_ie_eta_20261103_f.loss_curves import decisions
    at = 0 if task == "ie" else 300
    x = values[:, at:at+300]
    p, i = x[:, :, 0], x[:, :, 1]
    size = len(values)
    tol = 2 if model in ("vjepa2_vitg", "vjepa2_vitl", "videomaev2") else 1
    pa, hit, loc = decisions((i-p).transpose(2, 1, 0).reshape(x.shape[-1], -1), arrays["spans"][0 if task == "ie" else 1],
                             np.repeat(arrays["refs"][at:at+300], size), tol)
    pa, hit, loc = (v.reshape(300, size).T for v in (pa, hit, loc))
    pm, im = p.mean(2), i.mean(2)
    total = pm + im + 1e-12
    ranks = rankdata(np.concatenate((im/total, pm/total), axis=1), axis=1, method="average")
    auc = np.rint(2*(ranks[:, :300].sum(1) - 300*301/2)).astype(np.int64)
    valid = pa & (loc >= 0)
    count = valid.sum(1)
    mae = np.where(count > 0, (np.abs(loc-arrays["refs"][None, at:at+300])*valid).sum(1)/np.maximum(count, 1), 1e30)
    return hit.sum(1), pa.sum(1), auc, mae


def distinct(pool, limit):
    out = []
    for x in pool:
        if any(x["mode"] == y["mode"] and abs(x["eta"]-y["eta"]) <= max(1e-5, .025*max(x["eta"], y["eta"])) and abs(x["alpha"]-y["alpha"]) < .02 for y in out):
            continue
        out.append(x)
        if len(out) >= limit:
            break
    return out


def analyze_model(model):
    dest = c.HERE / "proposals" / (model + ".json")
    if dest.exists():
        return c.read(dest)
    settings, arrays = load_model(model)
    groups = defaultdict(lambda: defaultdict(list))
    for n, x in enumerate(settings):
        groups[(x["mode"], c.quality_key(x["parameters"]))][x["parameters"].get("alpha", 1.)].append(n)
    existing = c.existing_keys()
    candidates = {t: {role: [] for role in ("primary", "auc", "mae")} for t in ("sp", "ie")}
    diagnostics = []
    for (mode, qkey), groups_alpha in groups.items():
        alphas = sorted(groups_alpha)
        tasks = [t for t in ("sp", "ie") if model not in c.OBJECTS or qkey == c.quality_key(c.quality(model, t))]
        if len(alphas) < 2:
            continue
        for a0, a1 in zip(alphas, alphas[1:]):
            c.cpu_guard()
            curves, maxima = [], []
            for alpha in (a0, a1):
                ns = sorted(groups_alpha[alpha], key=lambda n: settings[n]["parameters"]["eta"])
                etas = np.asarray([0.] + [settings[n]["parameters"]["eta"] for n in ns])
                assert np.all(np.diff(etas) > 0)
                curves.append(PchipInterpolator(etas, np.concatenate((arrays["native"][None], arrays["losses"][ns])), axis=0))
                maxima.append(float(etas[-1]))
            high = min(maxima)
            eta = np.unique(np.r_[np.linspace(0., high, 769), np.geomspace(min(1e-5, high), high, 65)])
            left, right = (curve(eta) for curve in curves)
            # Search resolution belongs to tuning, never the model or scorer.
            count_alpha = max(2, int(np.ceil((a1-a0)/.0125))+1)
            for alpha in np.linspace(a0, a1, count_alpha):
                weight = (alpha-a0)/(a1-a0)
                values = left + weight*(right-left)
                for task in tasks:
                    jh, pa, auc, mae = evaluate(values, task, arrays, model)
                    best = rank_actual(model, task)
                    baseline = c.source()["selected"][task][model]["metrics"][task]["native_baseline"]["overall"]
                    btol = "2" if model in ("vjepa2_vitg", "vjepa2_vitl", "videomaev2") else "1"
                    keep = (jh >= round(baseline["joint_hit"][btol]*300)) & (pa >= round(baseline["paired_accuracy"]*300))
                    orders = dict(primary=np.lexsort((mae, -auc, -pa, -jh)),
                                  auc=np.lexsort((mae, -pa, -jh, -auc)),
                                  mae=np.lexsort((-auc, -pa, -jh, mae)))
                    for role, order in orders.items():
                        added = 0
                        for n in order[:32]:
                            if eta[n] <= 0 or (role != "primary" and not keep[n]):
                                continue
                            key = (int(jh[n]), int(pa[n]), int(auc[n]), -float(mae[n]))
                            if role == "primary" and key <= best:
                                continue
                            q = c.quality(model, task)
                            j = c.job(model, mode, eta[n], alpha, q, task if model in c.OBJECTS else None,
                                      proposal_task=task, proposal_role=role, proposal_rank=list(key),
                                      proposal_kind="bounded PCHIP in eta, linear interpolation between measured alpha; actual full600 mandatory",
                                      measured_alpha_bracket=[a0, a1], measured_eta_ceiling=high, proposal_only=True)
                            if c.semantic_key(model, mode, j["eta"], j["alpha"], q) in existing:
                                continue
                            candidates[task][role].append(j)
                            added += 1
                            if added >= 3:
                                break
                c.status(c.HERE / "analysis_progress.json", dict(at=time.time(), stage="joint_parameter_proposals", model=model, mode=mode, alpha=float(alpha), alpha_bracket=[a0, a1]))
            diagnostics.append(dict(mode=mode, alpha_bracket=[a0, a1], eta_ceiling=high, alpha_samples=count_alpha, eta_samples=len(eta), quality=qkey))
            del left, right
    result = dict(at=time.time(), model=model, diagnostics=diagnostics, candidates={})
    for task in ("sp", "ie"):
        result["candidates"][task] = {}
        for role in ("primary", "auc", "mae"):
            pool = candidates[task][role]
            if role == "primary":
                pool.sort(key=lambda x: (tuple(-v for v in x["proposal_rank"]), x["config_id"]))
            elif role == "auc":
                pool.sort(key=lambda x: (-x["proposal_rank"][2], -x["proposal_rank"][0], -x["proposal_rank"][1], x["config_id"]))
            else:
                pool.sort(key=lambda x: (-x["proposal_rank"][3], -x["proposal_rank"][0], -x["proposal_rank"][1], x["config_id"]))
            result["candidates"][task][role] = distinct(pool, 24)
    c.save(dest, result)
    print("JOINT PROPOSALS", model, {t: {r: len(v) for r, v in roles.items()} for t, roles in result["candidates"].items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=c.ORDER)
    args = parser.parse_args()
    c.verify()
    import torch
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid
    torch.set_num_threads(2)
    guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        for model in ([args.model] if args.model else c.ORDER):
            analyze_model(model)
        assert not torch.cuda.is_initialized()
    finally:
        guard.remove()
    if not args.model:
        c.save(c.HERE / "ANALYSIS_COMPLETE.json", dict(at=time.time(), neural_forwards=0, CUDA_initialized=False, source_report_sha256=c.digest(c.SOURCE)))


if __name__ == "__main__":
    try:
        main()
    except c.ResourceBusy as error:
        c.save(c.HERE / "events" / f"analysis_pause_{time.time_ns()}.json", dict(at=time.time(), error=str(error)))
        raise SystemExit(75)
