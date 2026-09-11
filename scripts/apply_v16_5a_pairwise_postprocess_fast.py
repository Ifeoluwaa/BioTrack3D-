#!/usr/bin/env python3
"""
BioTrack3D++ V16.5a — FAST standalone association post-processor.

This reproduces the validated V16.5a pairwise association layer without
rerunning the neural edge model or rebuilding the expensive research CSVs.

WHY THIS VERSION IS FASTER
--------------------------
1. Candidate source/target ranks/counts are computed with NumPy group sorting
   instead of Python dict-of-lists sorting.
2. Past/future tracklet summaries are memoized ONCE PER NODE rather than
   recomputed for every candidate/blocker edge.
3. Pairwise model input is built as a compact NumPy matrix, not a 98-column
   Python dict/DataFrame per duel.
4. Only candidates satisfying p>=0.20 and source-rank OR target-rank<=5 are
   examined.
5. Progress/timing is printed for every stage and every video.

REPRODUCTION POLICY
-------------------
The graph state matches the original validated research script exactly:
  final_edges = predecode_selected edges + ACCEPTED V16.4a rescue edges
from the V16.4a division-audit CSV.

Inference decisions use NO ground truth.

Frozen V16.5a association policy:
  - pairwise model threshold 0.60
  - challenger must beat every valid blocker (minimum win probability)
  - do not alter existing predicted division sources
  - do not remove an edge from a predicted division source
  - protect accepted V16.4a rescue edges
  - greedy non-overlap independently PER VIDEO

Optional --reference-actions compares the freshly regenerated action universe
and model scores against the saved research action CSV.

Optional --evaluate runs the official metric.

With --expect-test8, exact expected video-aware result is:
  selected/applied actions = 2376
  edge TP/FP/FN = 5271/421/347
  adjusted edge J ~= 0.849010
  division TP/FP/FN = 7/2/5
  overall ~= 0.899010
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import tracksdata as td

if not hasattr(pl, "Float16"):
    pl.Float16 = pl.Float32


EDGE_FEATURES = [
    "prob","dist_um","source_head","source_rank","target_rank",
    "source_best_prob","target_best_prob",
    "prob_minus_source_best","prob_minus_target_best",
    "source_count","target_count","source_above_count","target_above_count",
    "source_history","target_future",
    "step_speed","prev_speed","next_speed",
    "pre_residual_last","post_residual_first",
    "forward_residual_mean","backward_residual_mean",
    "source_target_velocity_gap",
    "cos_prev_step","cos_step_next","cos_source_target_velocity",
    "speed_ratio_prev_step","speed_ratio_step_next",
    "source_velocity_variation","target_velocity_variation",
    "source_straightness","target_straightness",
]

PAIR_FEATURES = (
    ["same_source_conflict", "same_target_conflict"]
    + [f"c_{f}" for f in EDGE_FEATURES]
    + [f"b_{f}" for f in EDGE_FEATURES]
    + [f"d_{f}" for f in EDGE_FEATURES]
)

NF = len(EDGE_FEATURES)
NPF = len(PAIR_FEATURES)


def secs(t0):
    return time.perf_counter() - t0


def finite_mean(x):
    vals = [float(v) for v in x if np.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def safe_cos(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return math.nan
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def speed_ratio(a, b):
    a = float(a)
    b = float(b)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-9 or b <= 1e-9:
        return math.nan
    return min(a, b) / max(a, b)


def velocity_variation(vels):
    if len(vels) < 2:
        return math.nan
    arr = np.asarray(vels, dtype=float)
    meanv = np.mean(arr, axis=0)
    return float(np.mean(np.linalg.norm(arr - meanv, axis=1)))


def straightness(vels):
    if len(vels) < 2:
        return math.nan
    vals = []
    for a, b in zip(vels[:-1], vels[1:]):
        c = safe_cos(a, b)
        if np.isfinite(c):
            vals.append(c)
    return finite_mean(vals)


def walk_past(node, incoming, limit=4):
    nodes = [int(node)]
    cur = int(node)
    seen = {cur}
    while len(nodes) <= limit:
        ps = incoming.get(cur, [])
        if len(ps) != 1:
            break
        nxt = int(ps[0])
        if nxt in seen:
            break
        nodes.append(nxt)
        seen.add(nxt)
        cur = nxt
    nodes.reverse()
    return nodes


def walk_future(node, outgoing, limit=4):
    nodes = [int(node)]
    cur = int(node)
    seen = {cur}
    while len(nodes) <= limit:
        ss = outgoing.get(cur, [])
        if len(ss) != 1:
            break
        nxt = int(ss[0])
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
        # Keep the exact arithmetic form used in the validated research script.
        out.append((xyz[b] - xyz[a]) * scale)
    return out


def vectorized_group_context(keys, prob, threshold):
    """
    Exact stable equivalent of:
      sorted(ids, key=lambda j: float(prob[j]), reverse=True)
    independently within each key group.

    Returns row-aligned arrays:
      rank, best_probability, total_count, count_above_threshold
    """
    n = len(keys)
    idx = np.arange(n, dtype=np.int64)

    # Primary key = node id, secondary = probability descending,
    # tertiary = original row index ascending to reproduce Python stable sort.
    order = np.lexsort((idx, -prob, keys))
    sk = keys[order]

    if n == 0:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.float32),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
        )

    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts_g = (ends - starts).astype(np.int32)

    rank_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, counts_g) + 1
    rank = np.empty(n, dtype=np.int32)
    rank[order] = rank_sorted.astype(np.int32)

    best_g = prob[order[starts]]
    best = np.empty(n, dtype=np.float32)
    best[order] = np.repeat(best_g, counts_g)

    count = np.empty(n, dtype=np.int32)
    count[order] = np.repeat(counts_g, counts_g)

    above_sorted = (prob[order] >= threshold).astype(np.int32)
    above_g = np.add.reduceat(above_sorted, starts).astype(np.int32)
    above = np.empty(n, dtype=np.int32)
    above[order] = np.repeat(above_g, counts_g)

    return rank, best, count, above


def prepare_video_state(
    video, pred, cache, accepted_rescues, scale, K, edge_threshold
):
    t0 = time.perf_counter()

    # Cache arrays exactly as in the research script.
    src = cache["source_id"].astype(np.int64, copy=False)
    tgt = cache["target_id"].astype(np.int64, copy=False)
    prob = cache["prob"].astype(np.float32, copy=False)
    dist = cache["dist_um"].astype(np.float32, copy=False)
    head = cache["source_division_head"].astype(np.float32, copy=False)
    pre = cache["predecode_selected"].astype(bool, copy=False)

    print(
        f"  candidates={len(src):,} predecode={int(pre.sum()):,}",
        flush=True,
    )

    # Exactly reconstruct the graph used in the original pairwise experiment.
    selected_idx = np.flatnonzero(pre)
    final_edges = {
        (int(src[i]), int(tgt[i]))
        for i in selected_idx
    }
    final_edges.update(accepted_rescues)

    if len(final_edges) != int(pred.num_edges()):
        raise RuntimeError(
            f"{video}: reconstructed final edges={len(final_edges)} "
            f"but pred.num_edges={pred.num_edges()}. "
            "Check the division-audit file."
        )

    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for s, t in final_edges:
        outgoing[s].append(t)
        incoming[t].append(s)

    ndf = pred.node_attrs(
        attr_keys=[K.NODE_ID, "t", "z", "y", "x"]
    ).to_dict(as_series=False)

    node_rows = []
    xyz = {}
    for nid, tt, z, y, x in zip(
        ndf[K.NODE_ID], ndf["t"], ndf["z"], ndf["y"], ndf["x"]
    ):
        nid = int(nid)
        node_rows.append({
            "node_id": nid,
            "t": int(tt),
            "z": float(z),
            "y": float(y),
            "x": float(x),
        })
        xyz[nid] = np.asarray(
            [float(z), float(y), float(x)],
            dtype=float,
        )

    print(f"  graph/node preload: {secs(t0):.1f}s", flush=True)

    t1 = time.perf_counter()
    srank, sbest, scount, sabove = vectorized_group_context(
        src, prob, edge_threshold
    )
    trank, tbest, tcount, tabove = vectorized_group_context(
        tgt, prob, edge_threshold
    )
    print(f"  vectorized ranks/context: {secs(t1):.1f}s", flush=True)

    t2 = time.perf_counter()

    # Blocker lookup only needs p>=threshold cache edges.
    eligible_prob_idx = np.flatnonzero(prob >= edge_threshold)
    pair_idx = {
        (int(src[i]), int(tgt[i])): int(i)
        for i in eligible_prob_idx
    }

    # Memoized trajectory summaries. Each source history / target future is
    # walked at most once even if it participates in many candidates.
    past_cache = {}
    future_cache = {}
    edge_cache = {}

    def past_summary(s):
        s = int(s)
        if s in past_cache:
            return past_cache[s]

        nodes = walk_past(s, incoming, 4)
        vels = node_velocities(nodes, xyz, scale)

        last = vels[-1] if vels else None
        mean = np.mean(vels, axis=0) if vels else None
        speed = float(np.linalg.norm(last)) if last is not None else math.nan

        out = {
            "history": len(vels),
            "last": last,
            "mean": mean,
            "speed": speed,
            "variation": velocity_variation(vels),
            "straightness": straightness(vels),
        }
        past_cache[s] = out
        return out

    def future_summary(t):
        t = int(t)
        if t in future_cache:
            return future_cache[t]

        nodes = walk_future(t, outgoing, 4)
        vels = node_velocities(nodes, xyz, scale)

        first = vels[0] if vels else None
        mean = np.mean(vels, axis=0) if vels else None
        speed = float(np.linalg.norm(first)) if first is not None else math.nan

        out = {
            "future": len(vels),
            "first": first,
            "mean": mean,
            "speed": speed,
            "variation": velocity_variation(vels),
            "straightness": straightness(vels),
        }
        future_cache[t] = out
        return out

    def edge_values(i):
        i = int(i)
        if i in edge_cache:
            return edge_cache[i]

        s = int(src[i])
        t = int(tgt[i])
        p = float(prob[i])

        ps = past_summary(s)
        fs = future_summary(t)

        if s in xyz and t in xyz:
            step = (xyz[t] - xyz[s]) * scale
            step_speed = float(np.linalg.norm(step))
            step_ok = bool(np.all(np.isfinite(step)))
        else:
            step = np.asarray([math.nan, math.nan, math.nan], dtype=float)
            step_speed = math.nan
            step_ok = False

        prev_last = ps["last"]
        next_first = fs["first"]
        prev_mean = ps["mean"]
        next_mean = fs["mean"]

        prev_speed = ps["speed"]
        next_speed = fs["speed"]

        pre_resid = (
            float(np.linalg.norm(step - prev_last))
            if prev_last is not None and step_ok
            else math.nan
        )
        post_resid = (
            float(np.linalg.norm(next_first - step))
            if next_first is not None and step_ok
            else math.nan
        )
        fwd_resid = (
            float(np.linalg.norm(step - prev_mean))
            if prev_mean is not None and step_ok
            else math.nan
        )
        back_resid = (
            float(np.linalg.norm(step - next_mean))
            if next_mean is not None and step_ok
            else math.nan
        )
        vel_gap = (
            float(np.linalg.norm(prev_mean - next_mean))
            if prev_mean is not None and next_mean is not None
            else math.nan
        )

        vals = np.asarray([
            p,
            float(dist[i]),
            float(head[i]),
            float(srank[i]),
            float(trank[i]),
            float(sbest[i]),
            float(tbest[i]),
            p - float(sbest[i]),
            p - float(tbest[i]),
            float(scount[i]),
            float(tcount[i]),
            float(sabove[i]),
            float(tabove[i]),
            float(ps["history"]),
            float(fs["future"]),
            step_speed,
            prev_speed,
            next_speed,
            pre_resid,
            post_resid,
            fwd_resid,
            back_resid,
            vel_gap,
            (
                safe_cos(prev_last, step)
                if prev_last is not None and step_ok
                else math.nan
            ),
            (
                safe_cos(step, next_first)
                if next_first is not None and step_ok
                else math.nan
            ),
            (
                safe_cos(prev_mean, next_mean)
                if prev_mean is not None and next_mean is not None
                else math.nan
            ),
            speed_ratio(prev_speed, step_speed),
            speed_ratio(step_speed, next_speed),
            float(ps["variation"]),
            float(fs["variation"]),
            float(ps["straightness"]),
            float(fs["straightness"]),
        ], dtype=np.float64)

        if len(vals) != NF:
            raise AssertionError((len(vals), NF))

        edge_cache[i] = vals
        return vals

    print(f"  blocker lookup setup: {secs(t2):.1f}s", flush=True)

    return {
        "src": src,
        "tgt": tgt,
        "prob": prob,
        "dist": dist,
        "head": head,
        "pre": pre,
        "srank": srank,
        "trank": trank,
        "final_edges": final_edges,
        "accepted_rescues": accepted_rescues,
        "outgoing": outgoing,
        "incoming": incoming,
        "pair_idx": pair_idx,
        "edge_values": edge_values,
        "node_rows": node_rows,
        "past_cache": past_cache,
        "future_cache": future_cache,
        "edge_cache": edge_cache,
    }


def build_and_score_actions(
    video, state, model, edge_threshold, rank_limit
):
    t0 = time.perf_counter()

    src = state["src"]
    tgt = state["tgt"]
    prob = state["prob"]
    srank = state["srank"]
    trank = state["trank"]

    final_edges = state["final_edges"]
    accepted_rescues = state["accepted_rescues"]
    outgoing = state["outgoing"]
    incoming = state["incoming"]
    pair_idx = state["pair_idx"]
    edge_values = state["edge_values"]

    # Aggressive inference-safe prefilter before entering Python candidate loop.
    eligible = np.flatnonzero(
        (prob >= edge_threshold)
        & ((srank <= rank_limit) | (trank <= rank_limit))
    )

    print(
        f"  rank/prob eligible candidates={len(eligible):,}",
        flush=True,
    )

    actions = []
    X_rows = []
    duel_action_idx = []

    for i in eligible:
        i = int(i)
        s = int(src[i])
        t = int(tgt[i])
        e = (s, t)

        if e in final_edges:
            continue

        # Candidate source is already a predicted division.
        if len(outgoing.get(s, [])) > 1:
            continue

        src_blockers = [
            (s, int(x))
            for x in outgoing.get(s, [])
            if int(x) != t
        ]
        tgt_blockers = [
            (int(y), t)
            for y in incoming.get(t, [])
            if int(y) != s
        ]
        blockers = list(dict.fromkeys(src_blockers + tgt_blockers))

        if not blockers:
            continue

        # Protect any predicted division edge.
        if any(
            len(outgoing.get(bs, [])) > 1
            for bs, _ in blockers
        ):
            continue

        # Exact original policy: accepted V16.4a rescue edges cannot be removed.
        if any(b in accepted_rescues for b in blockers):
            continue

        cf = edge_values(i)

        valid = []
        blocker_indices = []
        for b in blockers:
            bi = pair_idx.get(b)
            if bi is None:
                continue
            valid.append(b)
            blocker_indices.append(int(bi))

        if not valid:
            continue

        aidx = len(actions)
        actions.append({
            "video": video,
            "source_id": s,
            "target_id": t,
            "candidate_prob": float(cf[0]),
            "source_rank": float(cf[3]),
            "target_rank": float(cf[4]),
            "n_blockers": len(valid),
            "source_block_target": (
                src_blockers[0][1]
                if len(src_blockers) == 1
                else math.nan
            ),
            "target_block_source": (
                tgt_blockers[0][0]
                if len(tgt_blockers) == 1
                else math.nan
            ),
        })

        for b, bi in zip(valid, blocker_indices):
            bf = edge_values(bi)
            same_source = 1.0 if b[0] == s else 0.0
            same_target = 1.0 - same_source

            diff = np.where(
                np.isfinite(cf) & np.isfinite(bf),
                cf - bf,
                np.nan,
            )

            x = np.empty(NPF, dtype=np.float64)
            x[0] = same_source
            x[1] = same_target
            x[2:2 + NF] = cf
            x[2 + NF:2 + 2 * NF] = bf
            x[2 + 2 * NF:] = diff

            X_rows.append(x)
            duel_action_idx.append(aidx)

    print(
        f"  built actions={len(actions):,} duels={len(X_rows):,} "
        f"in {secs(t0):.1f}s",
        flush=True,
    )

    if not actions or not X_rows:
        return pd.DataFrame(actions), 0

    t1 = time.perf_counter()
    X = np.vstack(X_rows)
    action_idx = np.asarray(duel_action_idx, dtype=np.int64)

    win_prob = model.predict_proba(X)[:, 1]

    mins = np.full(len(actions), np.inf, dtype=np.float64)
    sums = np.zeros(len(actions), dtype=np.float64)
    counts = np.zeros(len(actions), dtype=np.int32)

    np.minimum.at(mins, action_idx, win_prob)
    np.add.at(sums, action_idx, win_prob)
    np.add.at(counts, action_idx, 1)

    means = sums / counts

    df = pd.DataFrame(actions)
    df["combined_win_prob"] = mins
    df["mean_win_prob"] = means
    df["n_scored_blockers"] = counts

    print(
        f"  model scoring/aggregation: {secs(t1):.1f}s",
        flush=True,
    )
    print(
        f"  memoized source histories={len(state['past_cache']):,} "
        f"target futures={len(state['future_cache']):,} "
        f"edge feature vectors={len(state['edge_cache']):,}",
        flush=True,
    )

    return df, len(X_rows)


def greedy_one_video(actions, threshold):
    cand = actions[
        actions["combined_win_prob"] >= threshold
    ].sort_values("combined_win_prob", ascending=False)

    chosen = []
    touched = set()

    for idx, r in cand.iterrows():
        nodes = {
            int(r["source_id"]),
            int(r["target_id"]),
        }

        sbt = r.get("source_block_target", np.nan)
        tbs = r.get("target_block_source", np.nan)

        if pd.notna(sbt):
            nodes.add(int(sbt))
        if pd.notna(tbs):
            nodes.add(int(tbs))

        if touched & nodes:
            continue

        chosen.append(idx)
        touched |= nodes

    return cand.loc[chosen].copy()


def action_blockers(r):
    s = int(r["source_id"])
    t = int(r["target_id"])
    out = []

    sbt = r.get("source_block_target", np.nan)
    tbs = r.get("target_block_source", np.nan)

    if pd.notna(sbt):
        out.append((s, int(sbt)))
    if pd.notna(tbs):
        out.append((int(tbs), t))

    return list(dict.fromkeys(out))


def apply_actions(base_edges, actions):
    edges = set(base_edges)
    applied = 0
    skipped = 0

    for _, r in actions.iterrows():
        cand = (int(r["source_id"]), int(r["target_id"]))
        bs = action_blockers(r)

        if cand in edges:
            skipped += 1
            continue
        if not bs or any(b not in edges for b in bs):
            skipped += 1
            continue

        for b in bs:
            edges.remove(b)
        edges.add(cand)
        applied += 1

    return edges, applied, skipped


def build_output_graph(node_rows, edges, cache):
    graph = td.graph.InMemoryGraph()

    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    new_ids = graph.bulk_add_nodes([
        {
            "t": int(r["t"]),
            "z": float(r["z"]),
            "y": float(r["y"]),
            "x": float(r["x"]),
        }
        for r in node_rows
    ])

    old_to_new = {
        int(r["node_id"]): int(new_id)
        for r, new_id in zip(node_rows, new_ids)
    }

    src = cache["source_id"].astype(np.int64, copy=False)
    tgt = cache["target_id"].astype(np.int64, copy=False)
    prob = cache["prob"].astype(np.float32, copy=False)
    dist = cache["dist_um"].astype(np.float32, copy=False)

    cache_map = {
        (int(src[i]), int(tgt[i])): (
            float(prob[i]), float(dist[i])
        )
        for i in range(len(src))
    }

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)

        rows = []
        for s, t in sorted(edges):
            p, d = cache_map.get((s, t), (0.0, 0.0))
            rows.append({
                "source_id": old_to_new[int(s)],
                "target_id": old_to_new[int(t)],
                "edge_prob": p,
                "edge_dist": d,
            })

        graph.bulk_add_edges(rows)

    return graph


def compare_reference(video, fresh, reference, score_tol):
    ref = reference[
        reference["video"].astype(str) == str(video)
    ].copy()

    fk = set(zip(
        fresh["source_id"].astype(int),
        fresh["target_id"].astype(int),
    ))
    rk = set(zip(
        ref["source_id"].astype(int),
        ref["target_id"].astype(int),
    ))

    common = fresh.merge(
        ref[["source_id", "target_id", "combined_win_prob"]],
        on=["source_id", "target_id"],
        how="inner",
        suffixes=("_fresh", "_ref"),
    )

    if len(common):
        diff = np.abs(
            common["combined_win_prob_fresh"].to_numpy(float)
            - common["combined_win_prob_ref"].to_numpy(float)
        )
        maxdiff = float(diff.max())
        meandiff = float(diff.mean())
    else:
        maxdiff = math.nan
        meandiff = math.nan

    out = {
        "video": video,
        "fresh_actions": len(fresh),
        "reference_actions": len(ref),
        "missing": len(rk - fk),
        "extra": len(fk - rk),
        "common": len(common),
        "max_score_abs_diff": maxdiff,
        "mean_score_abs_diff": meandiff,
    }

    ok = (
        out["missing"] == 0
        and out["extra"] == 0
        and (
            not np.isfinite(maxdiff)
            or maxdiff <= score_tol
        )
    )
    return out, ok


def accepted_rescues_for(video, audit):
    av = audit[
        (audit["video"].astype(str) == str(video))
        & (audit["decision"].astype(str) == "accepted")
    ]

    return {
        (int(r["parent"]), int(r["candidate_child"]))
        for _, r in av.iterrows()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--division-audit", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--reference-actions", type=Path, default=None)
    ap.add_argument("--strict-reference", action="store_true")
    ap.add_argument("--reference-score-tol", type=float, default=1e-10)

    ap.add_argument("--edge-threshold", type=float, default=0.20)
    ap.add_argument("--rank-limit", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None)

    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--expect-test8", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))

    from evaluate import _load_graph, _read_scale, _read_estimated_n_total
    from tracking_cellmot.io import save_graph
    from tracking_cellmot.metrics import (
        evaluate as compute_metric,
        _evaluate_matched_graph,
        node_recall as node_recall_fn,
        per_sample_metrics,
        summarise,
    )

    K = td.DEFAULT_ATTR_KEYS

    bundle = joblib.load(args.model)
    model = bundle["model"]

    if list(bundle.get("features", [])) != PAIR_FEATURES:
        raise RuntimeError(
            "Model feature list does not match validated V16.5a."
        )

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle["threshold"])
    )

    div_audit = pd.read_csv(
        args.division_audit,
        float_precision="round_trip",
    )

    reference = None
    if args.reference_actions is not None:
        reference = pd.read_csv(
            args.reference_actions,
            float_precision="round_trip",
        )

    pred_files = {p.stem: p for p in args.pred_dir.glob("*.geff")}
    cache_files = {p.stem: p for p in args.candidate_dir.glob("*.npz")}
    videos = sorted(set(pred_files) & set(cache_files))

    if not videos:
        raise RuntimeError(
            "No matching prediction GEFF / candidate NPZ files found."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== V16.5a FAST STANDALONE POST-PROCESSOR ===")
    print(f"videos: {len(videos)}")
    print(f"edge threshold: {args.edge_threshold:.2f}")
    print(f"rank limit: {args.rank_limit}")
    print(f"pairwise threshold: {threshold:.2f}")
    print("accepted rescues: exact division-audit reconstruction")
    print("GT used for association decisions: NO")
    print("non-overlap: PER VIDEO")
    print(flush=True)

    total_t0 = time.perf_counter()
    total_selected = 0
    total_applied = 0
    ref_rows = []

    metric_rows = []
    edge_tp = edge_fp = edge_fn = 0

    for pos, video in enumerate(videos, 1):
        vt0 = time.perf_counter()
        print(f"[{pos}/{len(videos)}] {video}", flush=True)

        t = time.perf_counter()
        pred = _load_graph(pred_files[video])
        cache = np.load(cache_files[video])
        full_scale = _read_scale(args.data_dir, video)
        scale = np.asarray(full_scale, dtype=float)
        if scale.size == 4:
            scale = scale[1:]

        rescues = accepted_rescues_for(video, div_audit)
        print(
            f"  load files: {secs(t):.1f}s "
            f"accepted_rescues={len(rescues)}",
            flush=True,
        )

        state = prepare_video_state(
            video,
            pred,
            cache,
            rescues,
            scale,
            K,
            args.edge_threshold,
        )

        scored, nduels = build_and_score_actions(
            video,
            state,
            model,
            args.edge_threshold,
            args.rank_limit,
        )

        if reference is not None:
            rr, ok = compare_reference(
                video,
                scored,
                reference,
                args.reference_score_tol,
            )
            ref_rows.append(rr)
            print(
                "  reference universe: "
                f"fresh/ref={rr['fresh_actions']:,}/{rr['reference_actions']:,} "
                f"missing={rr['missing']} extra={rr['extra']} "
                f"maxdiff={rr['max_score_abs_diff']:.12g}",
                flush=True,
            )
            if args.strict_reference and not ok:
                raise RuntimeError(
                    f"{video}: reference reproduction failed. "
                    "Stopping before graph rewrite/evaluation."
                )

        selected = greedy_one_video(scored, threshold)
        modified_edges, applied, skipped = apply_actions(
            state["final_edges"],
            selected,
        )

        total_selected += len(selected)
        total_applied += applied

        print(
            f"  p>={threshold:.2f}: selected={len(selected):,} "
            f"applied={applied:,} skipped={skipped:,}",
            flush=True,
        )

        t = time.perf_counter()
        out_graph = build_output_graph(
            state["node_rows"],
            modified_edges,
            cache,
        )
        out_path = args.output_dir / f"{video}.geff"
        save_graph(out_graph, out_path)
        print(
            f"  graph rebuild/save: {secs(t):.1f}s",
            flush=True,
        )

        if args.evaluate:
            t = time.perf_counter()
            gt_path = args.data_dir / f"{video}.geff"
            gt = _load_graph(gt_path)

            result = compute_metric(
                out_graph,
                gt,
                scale=full_scale,
                max_distance=7.0,
            )

            recall = (
                node_recall_fn(out_graph, gt)
                if out_graph.num_edges() > 0 and out_graph.num_nodes() > 0
                else 0.0
            )

            metric_rows.append(
                per_sample_metrics(
                    result,
                    _read_estimated_n_total(gt_path),
                    recall,
                )
            )

            medf = _evaluate_matched_graph(out_graph, gt)
            md = medf.to_dict(as_series=False)

            matched = np.asarray(
                md[K.MATCHED_EDGE_MASK],
                dtype=bool,
            )
            valid = np.asarray(
                md["pred_valid"],
                dtype=bool,
            )

            vtp = int(matched.sum())
            vfp = int(valid.sum()) - vtp

            ge = gt.edge_attrs(
                attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]
            ).to_dict(as_series=False)
            vfn = len(ge[K.EDGE_SOURCE]) - vtp

            edge_tp += vtp
            edge_fp += vfp
            edge_fn += vfn

            print(
                f"  official edge TP/FP/FN="
                f"{vtp}/{vfp}/{vfn} "
                f"evaluation={secs(t):.1f}s",
                flush=True,
            )

        print(
            f"  VIDEO TOTAL: {secs(vt0):.1f}s",
            flush=True,
        )

    print("\n=== FAST POST-PROCESS SUMMARY ===")
    print(f"selected actions: {total_selected}")
    print(f"applied actions: {total_applied}")
    print(f"wall time: {secs(total_t0):.1f}s")
    print(f"output: {args.output_dir}")

    if reference is not None:
        rdf = pd.DataFrame(ref_rows)
        print("\n=== REFERENCE ACTION REPRODUCTION ===")
        print(rdf.to_string(index=False))

        print(
            "TOTAL fresh/reference actions: "
            f"{int(rdf.fresh_actions.sum())}/"
            f"{int(rdf.reference_actions.sum())}"
        )
        print(
            "TOTAL missing/extra: "
            f"{int(rdf.missing.sum())}/"
            f"{int(rdf.extra.sum())}"
        )

        finite = rdf["max_score_abs_diff"].dropna()
        if len(finite):
            print(
                f"max action-score abs diff: "
                f"{float(finite.max()):.12g}"
            )

    if args.evaluate:
        sm = summarise(metric_rows)
        raw_j = edge_tp / (edge_tp + edge_fp + edge_fn)

        print("\n=== OFFICIAL METRIC ===")
        print(
            f"edge TP/FP/FN={edge_tp}/{edge_fp}/{edge_fn} "
            f"rawJ={raw_j:.6f}"
        )
        print(
            f"adjusted edge J="
            f"{float(sm['adj_edge_jaccard']):.6f}"
        )
        print(
            f"division TP/FP/FN="
            f"{int(sm['division_tp'])}/"
            f"{int(sm['division_fp'])}/"
            f"{int(sm['division_fn'])} "
            f"J={float(sm['division_jaccard']):.6f}"
        )
        print(
            f"node recall={float(sm['node_recall']):.6f}"
        )
        print(
            f"OVERALL={float(sm['score']):.6f}"
        )

        if args.expect_test8:
            failures = []

            if len(videos) != 8:
                failures.append(
                    f"videos={len(videos)} expected=8"
                )
            if total_applied != 2376:
                failures.append(
                    f"actions={total_applied} expected=2376"
                )
            if (edge_tp, edge_fp, edge_fn) != (5271, 421, 347):
                failures.append(
                    f"edge={(edge_tp,edge_fp,edge_fn)} "
                    "expected=(5271,421,347)"
                )
            if (
                int(sm["division_tp"]),
                int(sm["division_fp"]),
                int(sm["division_fn"]),
            ) != (7, 2, 5):
                failures.append(
                    "division counts expected=(7,2,5)"
                )
            if abs(
                float(sm["adj_edge_jaccard"]) - 0.849010
            ) > 5e-6:
                failures.append(
                    f"adjJ={float(sm['adj_edge_jaccard']):.9f} "
                    "expected~=0.849010"
                )
            if abs(
                float(sm["score"]) - 0.899010
            ) > 5e-6:
                failures.append(
                    f"score={float(sm['score']):.9f} "
                    "expected~=0.899010"
                )

            if failures:
                print(
                    "\nV16.5a FAST END-TO-END TEST8 REPRODUCTION: FAIL"
                )
                for f in failures:
                    print("  -", f)
                raise RuntimeError(
                    "Fast integration did not reproduce validated TEST8."
                )

            print(
                "\nV16.5a FAST END-TO-END TEST8 REPRODUCTION: PASS"
            )


if __name__ == "__main__":
    main()
