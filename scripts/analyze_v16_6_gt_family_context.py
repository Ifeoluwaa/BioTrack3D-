#!/usr/bin/env python3
"""
BioTrack3D++ V16.6 research experiment:
GT-derived temporal family-context classifier.

Question
--------
Can inference-safe multi-frame lineage context distinguish a real 1->2 mitosis
from the specific failure mode we now care about: an ordinary 1->1 continuation
plus a nearby unrelated cell that looks like a plausible second daughter?

This experiment is intentionally GT-only and CPU-only:
- positives: every annotated GT division (out-degree == 2)
- hard negatives: ordinary GT parents (out-degree == 1) paired with their true
  continuation + a nearby unrelated node at t+1 that passes the broad V16.4a
  family generator geometry
- NO neural edge probabilities
- NO division-head probability
- NO candidate's true incoming-parent identity (that would leak the label)
- NO predictor changes
- NO threshold tuning

The learned model is compared against the existing 4-feature learned biology
pair score under grouped cross-validation by video. Passing this test is only
a prerequisite for deployment: GT tracks are cleaner than predicted tracks.

Outputs
-------
<prefix>_families.csv
<prefix>_cv_scored.csv
<prefix>_cv_summary.csv
<prefix>_feature_importance.csv
<prefix>_model.joblib
<prefix>_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

if not hasattr(pl, "Float16"):
    pl.Float16 = pl.Float32


# Frozen V16.4a geometry used ONLY to create plausible pseudo-fork negatives.
PARENT_MAX_UM = 11.0
SISTER_MIN_UM = 5.0
SISTER_MAX_UM = 15.3
MIDPOINT_MAX_UM = 6.0

# Hard-negative generation policy: deterministic, not tuned.
NEGATIVES_PER_PARENT = 3
HISTORY_LIMIT = 4
FUTURE_LIMIT = 4

# Fixed research classifier. No hyperparameter sweep.
MODEL_KWARGS = dict(
    learning_rate=0.04,
    max_iter=240,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=5.0,
    early_stopping=False,
    random_state=0,
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def learned_division_pair_score(
    angle_deg: float,
    pds: float,
    sister_dist_um: float,
    midpoint_error_um: float,
) -> float:
    """Exact frozen 4-feature logistic mitosis score."""
    z_angle = (angle_deg - 103.11636742361918) / 39.635689953206395
    z_pds = (pds - 0.2606685995216205) / 0.21502299096448316
    z_sister = (sister_dist_um - 9.773215468212301) / 2.2774296624182515
    z_mid = (midpoint_error_um - 2.3459422462624873) / 0.27467692920998715

    z = (
        -5.499524582405182
        + 1.7994280408871015 * z_angle
        + 1.332141158603924 * z_pds
        - 0.5896250937667038 * z_sister
        + 0.5635888501431774 * z_mid
    )
    return float(sigmoid(float(z)))


def safe_cos(a, b):
    if a is None or b is None:
        return math.nan
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return math.nan
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def finite_norm(v):
    if v is None:
        return math.nan
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def physical_vec(a, b, scale):
    return (np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) * scale


def physical_distance(a, b, scale):
    return finite_norm(physical_vec(a, b, scale))


def division_angle(parent, c1, c2, scale):
    v1 = physical_vec(parent, c1, scale)
    v2 = physical_vec(parent, c2, scale)
    cos = safe_cos(v1, v2)
    if not np.isfinite(cos):
        return 0.0
    return float(np.degrees(np.arccos(cos)))


def straightness(vels):
    if not vels:
        return math.nan
    path = sum(float(np.linalg.norm(v)) for v in vels)
    if path <= 1e-8:
        return math.nan
    net = float(np.linalg.norm(np.sum(np.asarray(vels), axis=0)))
    return net / path


def velocity_variation(vels):
    if len(vels) < 2:
        return math.nan
    a = np.asarray(vels, dtype=float)
    mean = np.mean(a, axis=0)
    return float(np.mean(np.linalg.norm(a - mean, axis=1)))


def build_adjacency(edges):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for s, t in edges:
        outgoing[int(s)].append(int(t))
        incoming[int(t)].append(int(s))
    for v in outgoing.values():
        v.sort()
    for v in incoming.values():
        v.sort()
    return outgoing, incoming


def walk_past(start, incoming, max_steps=HISTORY_LIMIT):
    nodes = [int(start)]
    cur = int(start)
    seen = {cur}
    for _ in range(max_steps):
        prevs = incoming.get(cur, [])
        if len(prevs) != 1:
            break
        prev = int(prevs[0])
        if prev in seen:
            break
        nodes.append(prev)
        seen.add(prev)
        cur = prev
    nodes.reverse()
    return nodes


def walk_future(start, outgoing, max_steps=FUTURE_LIMIT):
    nodes = [int(start)]
    cur = int(start)
    seen = {cur}
    for _ in range(max_steps):
        nxts = outgoing.get(cur, [])
        if len(nxts) != 1:
            break
        nxt = int(nxts[0])
        if nxt in seen:
            break
        nodes.append(nxt)
        seen.add(nxt)
        cur = nxt
    return nodes


def node_velocities(nodes, xyz, scale):
    out = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        if a not in xyz or b not in xyz:
            continue
        out.append(physical_vec(xyz[a], xyz[b], scale))
    return out


def child_forward_summary(child, parent_xyz, xyz, outgoing, scale):
    nodes = walk_future(child, outgoing)
    vels = node_velocities(nodes, xyz, scale)

    step = (
        physical_vec(parent_xyz, xyz[child], scale)
        if child in xyz else None
    )
    next_vel = vels[0] if vels else None

    return {
        "nodes": nodes,
        "forward_len": max(0, len(nodes) - 1),
        "step": step,
        "step_speed": finite_norm(step),
        "next_speed": finite_norm(next_vel),
        "post_residual": (
            finite_norm(next_vel - step)
            if step is not None and next_vel is not None
            else math.nan
        ),
        "velocity_variation": velocity_variation(vels),
        "straightness": straightness(vels),
    }


def sym_pair(prefix, a, b, row):
    vals = [float(a), float(b)]
    finite = [v for v in vals if np.isfinite(v)]
    if len(finite) == 2:
        row[f"{prefix}_min"] = min(finite)
        row[f"{prefix}_max"] = max(finite)
        row[f"{prefix}_mean"] = 0.5 * (finite[0] + finite[1])
        row[f"{prefix}_absdiff"] = abs(finite[0] - finite[1])
    elif len(finite) == 1:
        row[f"{prefix}_min"] = finite[0]
        row[f"{prefix}_max"] = finite[0]
        row[f"{prefix}_mean"] = finite[0]
        row[f"{prefix}_absdiff"] = math.nan
    else:
        row[f"{prefix}_min"] = math.nan
        row[f"{prefix}_max"] = math.nan
        row[f"{prefix}_mean"] = math.nan
        row[f"{prefix}_absdiff"] = math.nan


def family_features(parent, c1, c2, xyz, outgoing, incoming, scale):
    """Order-invariant features available from a predicted lineage graph."""
    if parent not in xyz or c1 not in xyz or c2 not in xyz:
        return None

    p = xyz[parent]
    a = xyz[c1]
    b = xyz[c2]

    v1 = physical_vec(p, a, scale)
    v2 = physical_vec(p, b, scale)
    d1 = finite_norm(v1)
    d2 = finite_norm(v2)
    sister = physical_distance(a, b, scale)
    midpoint = 0.5 * (a + b)
    mid_err = physical_distance(p, midpoint, scale)
    angle = division_angle(p, a, b, scale)
    pds = min(d1, d2) / max(d1, d2) if max(d1, d2) > 1e-8 else 0.0
    biology = learned_division_pair_score(angle, pds, sister, mid_err)

    past_nodes = walk_past(parent, incoming)
    past_vels = node_velocities(past_nodes, xyz, scale)
    prev_last = past_vels[-1] if past_vels else None
    prev_mean = np.mean(past_vels, axis=0) if past_vels else None

    f1 = child_forward_summary(c1, p, xyz, outgoing, scale)
    f2 = child_forward_summary(c2, p, xyz, outgoing, scale)

    row = {
        "parent_dist_min": min(d1, d2),
        "parent_dist_max": max(d1, d2),
        "parent_dist_absdiff": abs(d1 - d2),
        "sister_dist_um": sister,
        "midpoint_error_um": mid_err,
        "angle_deg": angle,
        "pds": pds,
        "biology_pair_score": biology,

        "parent_history_len": max(0, len(past_nodes) - 1),
        "parent_prev_speed": finite_norm(prev_last),
        "parent_mean_speed": finite_norm(prev_mean),
        "parent_velocity_variation": velocity_variation(past_vels),
        "parent_straightness": straightness(past_vels),

        # How well each proposed daughter follows the parent's recent motion.
        "parent_to_child_cos_min": math.nan,
        "parent_to_child_cos_max": math.nan,
        "parent_step_residual_min": math.nan,
        "parent_step_residual_max": math.nan,
        "parent_step_residual_absdiff": math.nan,

        # Family midpoint motion relative to recent parent motion.
        "midpoint_step_speed": finite_norm(0.5 * (v1 + v2)),
        "midpoint_prev_residual": (
            finite_norm(0.5 * (v1 + v2) - prev_last)
            if prev_last is not None else math.nan
        ),
        "midpoint_meanprev_residual": (
            finite_norm(0.5 * (v1 + v2) - prev_mean)
            if prev_mean is not None else math.nan
        ),
    }

    if prev_last is not None:
        coss = [safe_cos(prev_last, v1), safe_cos(prev_last, v2)]
        coss_f = [x for x in coss if np.isfinite(x)]
        if coss_f:
            row["parent_to_child_cos_min"] = min(coss_f)
            row["parent_to_child_cos_max"] = max(coss_f)

        r1 = finite_norm(v1 - prev_last)
        r2 = finite_norm(v2 - prev_last)
        row["parent_step_residual_min"] = min(r1, r2)
        row["parent_step_residual_max"] = max(r1, r2)
        row["parent_step_residual_absdiff"] = abs(r1 - r2)

    # Symmetric daughter continuation summaries.
    sym_pair("daughter_forward_len", f1["forward_len"], f2["forward_len"], row)
    sym_pair("daughter_step_speed", f1["step_speed"], f2["step_speed"], row)
    sym_pair("daughter_next_speed", f1["next_speed"], f2["next_speed"], row)
    sym_pair("daughter_post_residual", f1["post_residual"], f2["post_residual"], row)
    sym_pair(
        "daughter_velocity_variation",
        f1["velocity_variation"], f2["velocity_variation"], row
    )
    sym_pair(
        "daughter_straightness",
        f1["straightness"], f2["straightness"], row
    )

    # Multi-frame separation of the two proposed branches.
    # k=0 is the child frame itself; k=1..3 are later unique-successor frames.
    n1 = f1["nodes"]
    n2 = f2["nodes"]
    initial = sister
    for k in (1, 2, 3):
        if len(n1) > k and len(n2) > k and n1[k] in xyz and n2[k] in xyz:
            sep = physical_distance(xyz[n1[k]], xyz[n2[k]], scale)
            row[f"future_sister_dist_k{k}"] = sep
            row[f"future_sister_growth_k{k}"] = sep - initial
            row[f"future_sister_ratio_k{k}"] = (
                sep / initial if initial > 1e-8 else math.nan
            )
        else:
            row[f"future_sister_dist_k{k}"] = math.nan
            row[f"future_sister_growth_k{k}"] = math.nan
            row[f"future_sister_ratio_k{k}"] = math.nan

    # Compact-route geometry is included only as a descriptive feature.
    row["compact_geometry"] = int(
        7.0 <= sister <= 15.3
        and angle >= 115.0
        and pds >= 0.70
        and mid_err <= 2.60
    )

    return row


def load_graph_tables(graph, K):
    nd = graph.node_attrs(
        attr_keys=[K.NODE_ID, "t", "z", "y", "x"]
    ).to_dict(as_series=False)

    xyz = {}
    times = {}
    by_time = defaultdict(list)
    for nid, t, z, y, x in zip(
        nd[K.NODE_ID], nd["t"], nd["z"], nd["y"], nd["x"]
    ):
        nid = int(nid)
        t = int(t)
        xyz[nid] = np.asarray([float(z), float(y), float(x)], dtype=float)
        times[nid] = t
        by_time[t].append(nid)

    ed = graph.edge_attrs(
        attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]
    ).to_dict(as_series=False)
    edges = [
        (int(s), int(t))
        for s, t in zip(ed[K.EDGE_SOURCE], ed[K.EDGE_TARGET])
    ]
    outgoing, incoming = build_adjacency(edges)

    return xyz, times, by_time, outgoing, incoming


def build_rows_for_video(video, graph, scale, K):
    xyz, times, by_time, outgoing, incoming = load_graph_tables(graph, K)

    rows = []
    n_pos = 0
    n_neg = 0

    # Positive: exact GT divider.
    for parent, children in outgoing.items():
        if len(children) != 2:
            continue
        c1, c2 = map(int, children)
        if (
            times.get(c1) != times.get(parent, -10) + 1
            or times.get(c2) != times.get(parent, -10) + 1
        ):
            continue

        feat = family_features(parent, c1, c2, xyz, outgoing, incoming, scale)
        if feat is None:
            continue

        rows.append({
            "video": video,
            "label": 1,
            "family_type": "gt_division",
            "parent": parent,
            "child1": c1,
            "child2": c2,
            **feat,
        })
        n_pos += 1

    # Negative: ordinary GT continuation + nearby unrelated t+1 cell.
    # Use physical coordinates in the KD-tree so radius is in um.
    tree_by_time = {}
    ids_by_time = {}
    for t, ids in by_time.items():
        ids = list(map(int, ids))
        if not ids:
            continue
        pts = np.asarray([xyz[n] * scale for n in ids], dtype=float)
        tree_by_time[t] = cKDTree(pts)
        ids_by_time[t] = ids

    for parent, children in outgoing.items():
        if len(children) != 1:
            continue

        existing = int(children[0])
        pt = times.get(parent)
        if pt is None or times.get(existing) != pt + 1:
            continue
        if pt + 1 not in tree_by_time:
            continue

        q = xyz[parent] * scale
        idxs = tree_by_time[pt + 1].query_ball_point(q, r=PARENT_MAX_UM)

        candidates = []
        for j in idxs:
            cand = int(ids_by_time[pt + 1][j])
            if cand == existing:
                continue

            feat = family_features(
                parent, existing, cand, xyz, outgoing, incoming, scale
            )
            if feat is None:
                continue

            # Broad V16.4a family generator region.
            if feat["parent_dist_max"] > PARENT_MAX_UM:
                continue
            if not (SISTER_MIN_UM <= feat["sister_dist_um"] <= SISTER_MAX_UM):
                continue
            if feat["midpoint_error_um"] > MIDPOINT_MAX_UM:
                continue

            candidates.append((feat["biology_pair_score"], cand, feat))

        # Keep only the highest biology-scoring pseudo-forks per ordinary parent.
        # This deliberately focuses the experiment on hard negatives rather than
        # letting easy negatives dominate.
        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        for rank, (_, cand, feat) in enumerate(
            candidates[:NEGATIVES_PER_PARENT], start=1
        ):
            rows.append({
                "video": video,
                "label": 0,
                "family_type": "ordinary_plus_nearby",
                "parent": parent,
                "child1": existing,
                "child2": cand,
                "negative_rank_within_parent": rank,
                **feat,
            })
            n_neg += 1

    return rows, n_pos, n_neg


META_COLUMNS = {
    "video", "label", "family_type", "parent", "child1", "child2",
    "negative_rank_within_parent",
}


def feature_columns(df):
    return [
        c for c in df.columns
        if c not in META_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def balanced_weights(y):
    y = np.asarray(y, dtype=int)
    n = len(y)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    w = np.ones(n, dtype=float)
    if n1:
        w[y == 1] = n / (2.0 * n1)
    if n0:
        w[y == 0] = n / (2.0 * n0)
    return w


def topk_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    k = int(y.sum())
    if k <= 0:
        return math.nan
    order = np.argsort(-score, kind="mergesort")
    return float(y[order[:k]].mean())


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "topk_precision_k_equals_positives": topk_precision(y, score),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--expected-gt-divisions",
        type=int,
        default=151,
        help="Sanity check against the previously built mitosis atlas.",
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))

    from evaluate import _load_graph, _read_scale

    K = td.DEFAULT_ATTR_KEYS
    geffs = sorted(args.data_dir.glob("*.geff"))
    if not geffs:
        raise FileNotFoundError(f"No .geff files found in {args.data_dir}")

    print("=== V16.6 GT FAMILY-CONTEXT EXPERIMENT ===")
    print(f"GT videos: {len(geffs)}")
    print("GPU required: NO")
    print("neural probabilities used: NO")
    print("candidate true-parent identity used: NO")
    print(
        f"hard negatives: top {NEGATIVES_PER_PARENT} biology-scoring "
        "pseudo-forks per ordinary parent"
    )
    print()

    all_rows = []
    total_pos = 0
    total_neg = 0

    for i, geff in enumerate(geffs, 1):
        video = geff.stem
        graph = _load_graph(geff)
        full_scale = _read_scale(args.data_dir, video)
        scale = np.asarray(full_scale, dtype=float)
        if scale.size == 4:
            scale = scale[1:]

        rows, np_, nn_ = build_rows_for_video(video, graph, scale, K)
        all_rows.extend(rows)
        total_pos += np_
        total_neg += nn_

        if i % 20 == 0 or i == len(geffs):
            print(
                f"[{i}/{len(geffs)}] positives={total_pos} "
                f"hard_negatives={total_neg:,}",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No family examples generated")

    print("\n=== DATASET ===")
    print(f"rows: {len(df):,}")
    print(f"positives: {int(df.label.sum())}")
    print(f"negatives: {int((df.label == 0).sum()):,}")
    print(f"videos represented: {df.video.nunique()}")

    if int(df.label.sum()) != args.expected_gt_divisions:
        print(
            f"WARNING: expected {args.expected_gt_divisions} positive GT divisions "
            f"but generated {int(df.label.sum())}. Inspect annotation/time filtering."
        )
    else:
        print("GT DIVISION COUNT CHECK: PASS")

    feats = feature_columns(df)
    print(f"features: {len(feats)}")

    X = df[feats]
    y = df["label"].astype(int).to_numpy()
    groups = df["video"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=0,
    )

    oof = np.full(len(df), np.nan, dtype=float)
    fold = np.full(len(df), -1, dtype=int)
    fold_rows = []

    for k, (tr, va) in enumerate(cv.split(X, y, groups), start=1):
        model = HistGradientBoostingClassifier(**MODEL_KWARGS)
        model.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=balanced_weights(y[tr]),
        )
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        fold[va] = k

        m = metrics(y[va], p)
        base = metrics(y[va], df.iloc[va]["biology_pair_score"].to_numpy())

        fold_rows.append({
            "fold": k,
            "n_train": len(tr),
            "n_valid": len(va),
            "valid_positives": int(y[va].sum()),
            "valid_negatives": int((y[va] == 0).sum()),
            "context_auc": m["roc_auc"],
            "context_ap": m["average_precision"],
            "context_topk_precision": m["topk_precision_k_equals_positives"],
            "biology_auc": base["roc_auc"],
            "biology_ap": base["average_precision"],
            "biology_topk_precision": base["topk_precision_k_equals_positives"],
        })

        print(
            f"fold{k}: pos={int(y[va].sum())} neg={int((y[va]==0).sum()):,} "
            f"context AUC/AP={m['roc_auc']:.4f}/{m['average_precision']:.4f} "
            f"biology={base['roc_auc']:.4f}/{base['average_precision']:.4f}",
            flush=True,
        )

    if np.isnan(oof).any():
        raise RuntimeError("OOF scores incomplete")

    overall_context = metrics(y, oof)
    overall_biology = metrics(y, df["biology_pair_score"].to_numpy())

    print("\n=== GROUPED OOF RESULT ===")
    print(
        "Existing biology only: "
        f"AUC={overall_biology['roc_auc']:.6f} "
        f"AP={overall_biology['average_precision']:.6f} "
        f"topK_precision={overall_biology['topk_precision_k_equals_positives']:.6f}"
    )
    print(
        "New temporal family context: "
        f"AUC={overall_context['roc_auc']:.6f} "
        f"AP={overall_context['average_precision']:.6f} "
        f"topK_precision={overall_context['topk_precision_k_equals_positives']:.6f}"
    )
    print(
        "DELTA context - biology: "
        f"AUC={overall_context['roc_auc'] - overall_biology['roc_auc']:+.6f} "
        f"AP={overall_context['average_precision'] - overall_biology['average_precision']:+.6f} "
        f"topK={overall_context['topk_precision_k_equals_positives'] - overall_biology['topk_precision_k_equals_positives']:+.6f}"
    )

    # Hardest compact-like subset.
    compact_mask = (
        (df["label"].to_numpy() == 1)
        | (df["compact_geometry"].fillna(0).to_numpy() > 0)
    )
    yc = y[compact_mask]
    if len(np.unique(yc)) == 2:
        cm = metrics(yc, oof[compact_mask])
        cb = metrics(
            yc,
            df.loc[compact_mask, "biology_pair_score"].to_numpy()
        )
        print("\n=== COMPACT-HARD SUBSET ===")
        print(
            f"rows={int(compact_mask.sum()):,} positives={int(yc.sum())} "
            f"negatives={int((yc==0).sum()):,}"
        )
        print(
            f"biology AUC/AP={cb['roc_auc']:.6f}/{cb['average_precision']:.6f}"
        )
        print(
            f"context AUC/AP={cm['roc_auc']:.6f}/{cm['average_precision']:.6f}"
        )

    # Fit final research model on all GT families.
    final_model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    final_model.fit(X, y, sample_weight=balanced_weights(y))

    # Permutation importance on OOF-unavailable full-fit data is diagnostic only;
    # do not use it as a performance estimate.
    # Use a deterministic stratified-ish sample to keep CPU cost bounded.
    rng = np.random.default_rng(0)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    max_neg = min(len(neg_idx), max(1000, 8 * len(pos_idx)))
    neg_sample = rng.choice(neg_idx, size=max_neg, replace=False) if len(neg_idx) > max_neg else neg_idx
    imp_idx = np.r_[pos_idx, neg_sample]
    imp = permutation_importance(
        final_model,
        X.iloc[imp_idx],
        y[imp_idx],
        scoring="average_precision",
        n_repeats=5,
        random_state=0,
        n_jobs=1,
    )
    imp_df = pd.DataFrame({
        "feature": feats,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(str(prefix) + "_families.csv")
    scored_path = Path(str(prefix) + "_cv_scored.csv")
    summary_path = Path(str(prefix) + "_cv_summary.csv")
    importance_path = Path(str(prefix) + "_feature_importance.csv")
    model_path = Path(str(prefix) + "_model.joblib")
    metadata_path = Path(str(prefix) + "_metadata.json")

    df.to_csv(dataset_path, index=False)

    scored = df.copy()
    scored["cv_fold"] = fold
    scored["context_oof_score"] = oof
    scored.to_csv(scored_path, index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(summary_path, index=False)
    imp_df.to_csv(importance_path, index=False)
    joblib.dump(final_model, model_path)

    metadata = {
        "experiment": "V16.6 GT temporal family context",
        "positive_definition": "GT parent with exactly two t+1 children",
        "negative_definition": (
            "GT ordinary parent + true continuation + nearby unrelated t+1 node "
            "inside parent<=11um, sister 5-15.3um, midpoint<=6um; top3 per parent "
            "by frozen biology pair score"
        ),
        "negative_per_parent": NEGATIVES_PER_PARENT,
        "history_limit": HISTORY_LIMIT,
        "future_limit": FUTURE_LIMIT,
        "model": "HistGradientBoostingClassifier",
        "model_kwargs": MODEL_KWARGS,
        "features": feats,
        "n_rows": int(len(df)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_videos": int(df.video.nunique()),
        "cv_splits": int(args.n_splits),
        "biology_oof_like_baseline": overall_biology,
        "context_oof": overall_context,
        "warning": (
            "GT-domain CV is a hypothesis test, not deployment validation. "
            "Next step must score frozen current V16.4a/V16.5a TP/FP/FN events "
            "using predicted-graph features before changing the predictor."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\n=== TOP 20 CONTEXT FEATURES (diagnostic permutation importance) ===")
    print(imp_df.head(20).to_string(index=False))

    print("\n=== FILES WRITTEN ===")
    for p in (
        dataset_path, scored_path, summary_path,
        importance_path, model_path, metadata_path,
    ):
        print(p)

    print("\nNEXT DECISION RULE:")
    print(
        "Do NOT deploy from this result alone. "
        "If grouped CV materially beats the frozen biology baseline, "
        "score the current evaluator-exact 7 TP / 6 FP / 12 FN events next."
    )


if __name__ == "__main__":
    main()
