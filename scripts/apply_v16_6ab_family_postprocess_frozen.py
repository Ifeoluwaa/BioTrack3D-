#!/usr/bin/env python3
"""
BioTrack3D++ V16.6a + V16.6b frozen FAMILY postprocessor for deployment.

This is the production/inference counterpart of the held-out research tests.

Input graph:
    V16.5a postprocessed graph.

Frozen V16.6a:
    For an already-accepted V16.4a COMPACT rescue, score its current 1->2 family
    with the frozen full V16.6 family-context model. If score < 0.50, remove only
    the added candidate daughter edge. Established and close-sister rescues are
    untouched.

Frozen V16.6b:
    Consider only V16.4a rows rejected for low biology pair score that otherwise
    satisfy the established-like route:
        sister_in_atlas
        head >= 0.83
        existing edge >= 0.88
        candidate edge >= 0.40
        parent history >= 3
        existing forward >= 3
        candidate forward >= 3
    Recheck current topology after V16.6a. If the candidate is unclaimed, the
    parent has exactly the audited existing child, daughter branches are separate,
    and the frozen family score >= 0.50, add only parent->candidate.

Deployment model policy:
    The frozen full family model is trained on all labeled TRAIN families.
    A hidden test video is not part of that labeled training set, so full-model
    inference is the deployment analogue of the local leave-current-video-out
    research protocol.

No GT is read or used. No thresholds are tuned. V16.7 is not included.
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

if not hasattr(pl, "Float16"):
    pl.Float16 = pl.Float32

FAMILY_THRESHOLD = 0.50


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def node_sig(row):
    return (
        int(row["t"]),
        round(float(row["z"]), 9),
        round(float(row["y"]), 9),
        round(float(row["x"]), 9),
    )


def graph_tables(graph, K):
    nd = graph.node_attrs(
        attr_keys=[K.NODE_ID, "t", "z", "y", "x"]
    ).to_dict(as_series=False)

    rows = []
    by_id = {}
    sig_to_id = {}
    duplicate_sigs = set()

    for nid, t, z, y, x in zip(
        nd[K.NODE_ID], nd["t"], nd["z"], nd["y"], nd["x"]
    ):
        nid = int(nid)
        row = {
            "node_id": nid,
            "t": int(t),
            "z": float(z),
            "y": float(y),
            "x": float(x),
        }
        rows.append(row)
        by_id[nid] = row

        sig = node_sig(row)
        if sig in sig_to_id:
            duplicate_sigs.add(sig)
        else:
            sig_to_id[sig] = nid

    if duplicate_sigs:
        raise RuntimeError(
            f"Found {len(duplicate_sigs)} duplicate detection signatures."
        )

    ed = graph.edge_attrs(
        attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]
    ).to_dict(as_series=False)

    edges = set()
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for s, t in zip(ed[K.EDGE_SOURCE], ed[K.EDGE_TARGET]):
        s = int(s)
        t = int(t)
        edges.add((s, t))
        outgoing[s].append(t)
        incoming[t].append(s)

    for vals in outgoing.values():
        vals.sort()
    for vals in incoming.values():
        vals.sort()

    xyz = {
        int(r["node_id"]): np.asarray(
            [r["z"], r["y"], r["x"]], dtype=float
        )
        for r in rows
    }

    return rows, by_id, sig_to_id, xyz, edges, outgoing, incoming


def map_old_id(old_id, raw_by_id, final_sig_to_id, video):
    old_id = int(old_id)
    if old_id not in raw_by_id:
        raise RuntimeError(f"{video}: raw V16.4a node {old_id} not found.")
    sig = node_sig(raw_by_id[old_id])
    if sig not in final_sig_to_id:
        raise RuntimeError(
            f"{video}: cannot map raw V16.4a node {old_id} "
            f"by detection signature {sig}."
        )
    return int(final_sig_to_id[sig])


def branches_separate(existing, candidate, outgoing):
    a = list(map(int, outgoing.get(int(existing), [])))
    b = list(map(int, outgoing.get(int(candidate), [])))
    if a and b:
        return a[0] != b[0]
    return True


def hypothetical_adjacency(parent, candidate, outgoing, incoming):
    out2 = defaultdict(list, {k: list(v) for k, v in outgoing.items()})
    in2 = defaultdict(list, {k: list(v) for k, v in incoming.items()})

    if int(candidate) not in out2[int(parent)]:
        out2[int(parent)].append(int(candidate))
        out2[int(parent)].sort()
    if int(parent) not in in2[int(candidate)]:
        in2[int(candidate)].append(int(parent))
        in2[int(candidate)].sort()
    return out2, in2


def rebuild_graph(node_rows, edges):
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

    if edges:
        graph.bulk_add_edges([
            {
                "source_id": old_to_new[int(s)],
                "target_id": old_to_new[int(t)],
            }
            for s, t in sorted(edges)
        ])
    return graph


def score_family(model, features, feat):
    X = pd.DataFrame([{
        c: feat.get(c, math.nan)
        for c in features
    }])
    return float(model.predict_proba(X[features])[:, 1][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--raw-v16-4a-pred-dir", type=Path, required=True)
    ap.add_argument("--v16-5a-pred-dir", type=Path, required=True)
    ap.add_argument("--division-audit", type=Path, required=True)
    ap.add_argument("--family-model", type=Path, required=True)
    ap.add_argument("--family-metadata", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--actions-out", type=Path, default=None)
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))

    from analyze_v16_6_gt_family_context import family_features
    from evaluate import _load_graph, _read_scale
    from tracking_cellmot.io import save_graph

    K = td.DEFAULT_ATTR_KEYS

    metadata = json.loads(args.family_metadata.read_text())
    features = metadata.get("features")
    if not isinstance(features, list) or len(features) != 55:
        raise RuntimeError(
            "Frozen V16.6 metadata must contain the exact 55-feature list."
        )

    model = joblib.load(args.family_model)
    nfi = getattr(model, "n_features_in_", len(features))
    if int(nfi) != len(features):
        raise RuntimeError(
            f"Family model expects {nfi} features; metadata has {len(features)}."
        )

    audit = pd.read_csv(
        args.division_audit,
        float_precision="round_trip",
        low_memory=False,
    )

    raw_files = {
        p.stem: p for p in args.raw_v16_4a_pred_dir.glob("*.geff")
    }
    base_files = {
        p.stem: p for p in args.v16_5a_pred_dir.glob("*.geff")
    }
    videos = sorted(set(raw_files) & set(base_files))

    if not videos:
        raise RuntimeError(
            "No matching V16.4a/V16.5a GEFF files found."
        )
    if set(raw_files) != set(base_files):
        raise RuntimeError(
            "V16.4a and V16.5a video sets differ; refusing partial inference."
        )

    required = {
        "video","parent","existing_child","candidate_child","decision",
        "route_name","head_prob","existing_edge_prob",
        "candidate_edge_prob","pair_score","sister_in_atlas",
        "parent_history_len","existing_forward_len","candidate_forward_len",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise RuntimeError(
            "Division audit is missing frozen V16.6 fields: "
            + ", ".join(missing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_rows = []

    print("=== FROZEN V16.6a + V16.6b DEPLOYMENT POSTPROCESSOR ===")
    print(f"videos: {len(videos)}")
    print(f"family threshold: {FAMILY_THRESHOLD:.2f}")
    print("family feature order: exact frozen metadata")
    print("family model: frozen full TRAIN-family model")
    print("GT used: NO")
    print("V16.7 included: NO")
    print()

    low_pair_mask = (
        (audit["decision"].astype(str) == "reject_low_pair_score")
        & bool_series(audit["sister_in_atlas"])
        & (audit["head_prob"].astype(float) >= 0.83)
        & (audit["existing_edge_prob"].astype(float) >= 0.88)
        & (audit["candidate_edge_prob"].astype(float) >= 0.40)
        & (audit["parent_history_len"].astype(float) >= 3)
        & (audit["existing_forward_len"].astype(float) >= 3)
        & (audit["candidate_forward_len"].astype(float) >= 3)
    )
    low_pair_all = audit.loc[low_pair_mask].copy()

    for pos, video in enumerate(videos, 1):
        raw = _load_graph(raw_files[video])
        base = _load_graph(base_files[video])

        (
            _raw_rows, raw_by_id, _raw_sig, _raw_xyz,
            _raw_edges, _raw_out, _raw_in,
        ) = graph_tables(raw, K)

        (
            node_rows, _base_by_id, sig_to_id, xyz,
            base_edges, outgoing, incoming,
        ) = graph_tables(base, K)

        if len(raw_by_id) != len(sig_to_id):
            raise RuntimeError(
                f"{video}: raw/base node-count mismatch "
                f"{len(raw_by_id)} != {len(sig_to_id)}"
            )

        full_scale = _read_scale(args.data_dir, video)
        scale = np.asarray(full_scale, dtype=float)
        if scale.size == 4:
            scale = scale[1:]

        modified = set(base_edges)

        # --------------------------------------------------------------
        # V16.6a: compact-family veto. Scores use the supplied V16.5a
        # topology, matching the frozen research implementation.
        # --------------------------------------------------------------
        compact = audit[
            (audit["video"].astype(str) == str(video))
            & (audit["decision"].astype(str) == "accepted")
            & audit["route_name"].astype(str).str.contains(
                "compact", na=False
            )
        ].copy()

        n_compact_scored = 0
        n_veto = 0

        for _, r in compact.iterrows():
            parent = map_old_id(
                r["parent"], raw_by_id, sig_to_id, video
            )
            existing = map_old_id(
                r["existing_child"], raw_by_id, sig_to_id, video
            )
            candidate = map_old_id(
                r["candidate_child"], raw_by_id, sig_to_id, video
            )

            children = list(map(int, outgoing.get(parent, [])))
            score = math.nan
            decision = "skip_topology_mismatch"
            feat = {}

            if (
                len(children) == 2
                and existing in children
                and candidate in children
            ):
                feat0 = family_features(
                    parent,
                    children[0],
                    children[1],
                    xyz,
                    outgoing,
                    incoming,
                    scale,
                )
                if feat0 is None:
                    decision = "skip_unscoreable"
                else:
                    feat = feat0
                    score = score_family(model, features, feat)
                    n_compact_scored += 1

                    edge = (parent, candidate)
                    if score < FAMILY_THRESHOLD:
                        if edge in modified:
                            modified.remove(edge)
                            decision = "veto"
                            n_veto += 1
                        else:
                            decision = "skip_edge_missing"
                    else:
                        decision = "keep"

            action_rows.append({
                "stage": "V16.6a",
                "video": video,
                "raw_parent": int(r["parent"]),
                "raw_existing_child": int(r["existing_child"]),
                "raw_candidate_child": int(r["candidate_child"]),
                "mapped_parent": parent,
                "mapped_existing_child": existing,
                "mapped_candidate_child": candidate,
                "family_score": score,
                "decision": decision,
                **feat,
            })

        # Recompute topology after V16.6a vetoes before V16.6b.
        outgoing2, incoming2 = (
            defaultdict(list), defaultdict(list)
        )
        for s, t in modified:
            outgoing2[int(s)].append(int(t))
            incoming2[int(t)].append(int(s))
        for vals in outgoing2.values():
            vals.sort()
        for vals in incoming2.values():
            vals.sort()

        # --------------------------------------------------------------
        # V16.6b: exact established-like low-pair override.
        # --------------------------------------------------------------
        low_pair = low_pair_all[
            low_pair_all["video"].astype(str) == str(video)
        ].copy()

        n_added = 0

        for _, r in low_pair.iterrows():
            parent = map_old_id(
                r["parent"], raw_by_id, sig_to_id, video
            )
            existing = map_old_id(
                r["existing_child"], raw_by_id, sig_to_id, video
            )
            candidate = map_old_id(
                r["candidate_child"], raw_by_id, sig_to_id, video
            )

            source_children = list(
                map(int, outgoing2.get(parent, []))
            )
            candidate_incoming = list(
                map(int, incoming2.get(candidate, []))
            )

            topology_ok = (
                len(source_children) == 1
                and existing in source_children
                and len(candidate_incoming) == 0
                and branches_separate(
                    existing, candidate, outgoing2
                )
            )

            score = math.nan
            decision = "reject_topology"
            feat = {}

            if topology_ok:
                out_hyp, in_hyp = hypothetical_adjacency(
                    parent, candidate, outgoing2, incoming2
                )
                feat0 = family_features(
                    parent,
                    existing,
                    candidate,
                    xyz,
                    out_hyp,
                    in_hyp,
                    scale,
                )
                if feat0 is None:
                    decision = "reject_unscoreable"
                else:
                    feat = feat0
                    score = score_family(model, features, feat)
                    if score >= FAMILY_THRESHOLD:
                        edge = (parent, candidate)
                        if edge in modified:
                            decision = "skip_already_present"
                        else:
                            modified.add(edge)
                            outgoing2[parent].append(candidate)
                            outgoing2[parent].sort()
                            incoming2[candidate].append(parent)
                            incoming2[candidate].sort()
                            decision = "accepted_add"
                            n_added += 1
                    else:
                        decision = "reject_family_below_0p50"

            action_rows.append({
                "stage": "V16.6b",
                "video": video,
                "raw_parent": int(r["parent"]),
                "raw_existing_child": int(r["existing_child"]),
                "raw_candidate_child": int(r["candidate_child"]),
                "mapped_parent": parent,
                "mapped_existing_child": existing,
                "mapped_candidate_child": candidate,
                "family_score": score,
                "decision": decision,
                "parent_outdegree_before": len(source_children),
                "candidate_indegree_before": len(candidate_incoming),
                **feat,
            })

        out_graph = rebuild_graph(node_rows, modified)
        save_graph(out_graph, args.output_dir / f"{video}.geff")

        print(
            f"[{pos:02d}/{len(videos)}] {video}: "
            f"compact={len(compact)} scored={n_compact_scored} "
            f"vetoed={n_veto}; "
            f"low_pair={len(low_pair)} added={n_added}",
            flush=True,
        )

    actions_out = (
        args.actions_out
        if args.actions_out is not None
        else args.output_dir.parent
        / f"{args.output_dir.name}_family_actions.csv"
    )
    actions_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(action_rows).to_csv(actions_out, index=False)

    print("\n=== COMPLETE ===")
    print("final graphs:", args.output_dir)
    print("family actions:", actions_out)


if __name__ == "__main__":
    main()
