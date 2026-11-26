"""Pure NumPy scoring; no adapters, weights, tensor predictions, or GPU calls."""
import numpy as np

TYPES = {'sp': ('sudden_disappearance', 'identity_change', 'occluded_wrong_reappearance'),
         'ie': ('motion_anomaly', 'penetration', 'contact_response')}


def calculate(possible, impossible, spans, tau):
    p, i = np.asarray(possible, dtype=np.float64), np.asarray(impossible, dtype=np.float64)
    if p.ndim == 2:
        p, i = p[None], i[None]
    assert p.ndim == 3 and p.shape == i.shape
    assert np.isfinite(p).all() and np.isfinite(i).all() and (p >= 0).all() and (i >= 0).all()
    spans = np.asarray(spans, dtype=np.int64)
    if spans.ndim == 2:
        spans = np.broadcast_to(spans, (p.shape[1], *spans.shape))
    tau = np.asarray(tau, dtype=np.int64)
    assert spans.shape == (p.shape[1], p.shape[2], 2) and tau.shape == (p.shape[1],)
    assert (spans[:, :, 1] >= spans[:, :, 0]).all()
    assert (spans[:, 1:, 0] > spans[:, :-1, 1]).all()
    match = (spans[:, :, 0] <= tau[:, None]) & (tau[:, None] <= spans[:, :, 1])
    assert (match.sum(1) == 1).all(), 'Every annotation must belong to exactly one original error span'
    index = match.argmax(1)
    onset_p = np.take_along_axis(p, index[None, :, None], axis=2)[:, :, 0]
    onset_i = np.take_along_axis(i, index[None, :, None], axis=2)[:, :, 0]
    pm, im = p.mean(2), i.mean(2)
    pa = im > pm
    vsa = onset_i > onset_p
    joint = pa & vsa
    # Original AUC normalization and original conditional localization MAE.
    score_p = pm / (pm + im + 1e-12)
    score_i = im / (pm + im + 1e-12)
    signal = np.log((i + 1e-12) / (p + 1e-12))
    located = np.zeros(pa.shape, bool)
    first = np.zeros(pa.shape, np.int64)
    run_length = np.zeros(pa.shape, np.int64)
    run_start = np.zeros(pa.shape, np.int64)
    previous_frame = np.full((p.shape[1],), np.iinfo(np.int64).min, np.int64)
    # Existing spans have constant width at each step; reject unexpected grids.
    for t in range(p.shape[2]):
        width = spans[:, t, 1] - spans[:, t, 0] + 1
        assert (width == width[0]).all()
        positive = signal[:, :, t] > 0.
        for offset in range(int(width[0])):
            frame = spans[:, t, 0] + offset
            contiguous = (frame == previous_frame + 1)[None, :]
            starting = positive & ((run_length == 0) | ~contiguous)
            run_start = np.where(starting, frame[None, :], run_start)
            run_length = np.where(positive, np.where(contiguous, run_length + 1, 1), 0)
            newly = (run_length >= 2) & ~located
            first = np.where(newly, run_start, first)
            located |= newly
            previous_frame = frame
    error = np.abs(first - tau[None, :])
    return dict(pairacc=pa, vsa=vsa, joint=joint, pair_tie=im == pm, vsa_tie=onset_i == onset_p,
                score_p=score_p, score_i=score_i, located=located, t_star=first,
                absolute_localization_error=error, mae_members=pa & located,
                onset_possible=onset_p, onset_impossible=onset_i, onset_error_index=index)


def pooled_auc(positive, negative):
    negative = np.sort(np.asarray(negative, dtype=np.float64))
    positive = np.asarray(positive, dtype=np.float64)
    below = np.searchsorted(negative, positive, side='left')
    above = np.searchsorted(negative, positive, side='right')
    return float((below.sum() + .5 * (above - below).sum()) / (positive.size * negative.size))


def aggregate(flags, select, k=0):
    ids = np.asarray(select, dtype=np.int64)
    assert len(ids)
    members = flags['mae_members'][k, ids]
    mae = float(flags['absolute_localization_error'][k, ids][members].mean()) if members.any() else None
    return dict(num_pairs=int(len(ids)), pairacc=float(flags['pairacc'][k, ids].mean()),
                auc=pooled_auc(flags['score_i'][k, ids], flags['score_p'][k, ids]),
                localization_mae=mae, localization_mae_denominator=int(members.sum()),
                detection_coverage=float(flags['located'][k, ids].mean()),
                vsa=float(flags['vsa'][k, ids].mean()), joint_accuracy=float(flags['joint'][k, ids].mean()),
                pairacc_count=int(flags['pairacc'][k, ids].sum()), vsa_count=int(flags['vsa'][k, ids].sum()),
                joint_count=int(flags['joint'][k, ids].sum()),
                pair_ties=int(flags['pair_tie'][k, ids].sum()), vsa_ties=int(flags['vsa_tie'][k, ids].sum()))


def metrics(flags, records, k=0):
    result = {}
    for task in ('sp', 'ie'):
        ids = [n for n, r in enumerate(records) if r['benchmark'] == task]
        assert len(ids) == 300
        groups = {cat: [n for n in ids if records[n]['category'] == cat] for cat in TYPES[task]}
        assert all(len(v) == 100 for v in groups.values())
        result[task] = dict(overall=aggregate(flags, ids, k),
                            by_type={cat: aggregate(flags, v, k) for cat, v in groups.items()})
    return result
