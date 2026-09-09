#!/usr/bin/env python
"""Label a V16.2 rescue audit using the competition evaluator's GT matching.

This does NOT rerun inference. It reopens the already-saved prediction GEFFs,
uses the official metric to match prediction nodes/edges to GT, and annotates
rescue proposals with:
  - parent/existing/candidate matched GT node ids
  - whether the proposed candidate edge is an exact GT edge
  - whether the full proposed 1->2 family is an exact GT division pair
  - for edges actually added to the saved graph: TP / FP / ignored under the
    official edge metric validity rules

Run from the BioTrack3D- repository root, for example:

python /kaggle/working/analyze_v16_2_rescues.py \
  --repo /kaggle/working/BioTrack3D- \
  --audit /kaggle/working/v16_2_rescue_audit_8.csv \
  --pred-dir /kaggle/working/BioTrack3D-/predictions/unknown/unet_transformer/split_0 \
  --output /kaggle/working/v16_2_rescue_audit_8_labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td


def _safe_int(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        value = int(value)
    except Exception:
        return None
    return None if value == -1 else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=7.0)
    args = parser.parse_args()

    repo = args.repo.resolve()
    scripts_dir = repo / "scripts"
    src_dir = repo / "src"
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(src_dir))

    from dataspec import DATASET_PATH
    from evaluate import _load_graph, _read_scale
    from tracking_cellmot.metrics import (
        evaluate as compute_metric,
        _evaluate_matched_graph,
    )

    audit = pd.read_csv(args.audit)
    if "video" not in audit.columns:
        raise ValueError("Audit CSV must contain a 'video' column")

    # Columns populated below.
    for col in (
        "parent_gt",
        "existing_child_gt",
        "candidate_child_gt",
        "gt_parent_outdegree",
    ):
        audit[col] = pd.Series([pd.NA] * len(audit), dtype="Int64")

    audit["candidate_edge_is_exact_gt"] = False
    audit["existing_edge_is_exact_gt"] = False
    audit["proposal_is_exact_gt_division_pair"] = False
    audit["added_edge_metric_status"] = "not_added"

    K = td.DEFAULT_ATTR_KEYS
    data_dir = Path(DATASET_PATH)

    per_video_summaries = []

    for video in audit["video"].dropna().astype(str).unique():
        pred_path = args.pred_dir / f"{video}.geff"
        gt_path = data_dir / f"{video}.geff"

        if not pred_path.exists():
            print(f"[SKIP] missing prediction: {pred_path}")
            continue
        if not gt_path.exists():
            print(f"[SKIP] missing GT: {gt_path}")
            continue

        pred_graph = _load_graph(pred_path)
        gt_graph = _load_graph(gt_path)
        scale = _read_scale(data_dir, video)

        # Mutates pred_graph by writing matched-node / matched-edge attributes.
        result = compute_metric(
            pred_graph,
            gt_graph,
            scale=scale,
            max_distance=args.max_distance,
        )

        # Predicted node -> matched GT node.
        node_df = pred_graph.node_attrs(
            attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]
        ).to_dict(as_series=False)
        pred_to_gt = {}
        for pred_id, gt_id in zip(node_df[K.NODE_ID], node_df[K.MATCHED_NODE_ID]):
            gt_id = _safe_int(gt_id)
            if gt_id is not None:
                pred_to_gt[int(pred_id)] = gt_id

        # Exact GT edge set and GT out-degree.
        gt_edges_df = gt_graph.edge_attrs(
            attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]
        ).to_dict(as_series=False)
        gt_edges = set()
        gt_outdegree = Counter()
        for source, target in zip(
            gt_edges_df[K.EDGE_SOURCE],
            gt_edges_df[K.EDGE_TARGET],
        ):
            source = int(source)
            target = int(target)
            gt_edges.add((source, target))
            gt_outdegree[source] += 1

        # Apply the same edge validity/dedup/out-degree handling used by the metric.
        evaluated_edges = _evaluate_matched_graph(pred_graph, gt_graph)
        edge_dict = evaluated_edges.to_dict(as_series=False)

        # Map final predicted (source,target) to metric status.
        final_edge_status = {}
        for source, target, matched, valid in zip(
            edge_dict[K.EDGE_SOURCE],
            edge_dict[K.EDGE_TARGET],
            edge_dict[K.MATCHED_EDGE_MASK],
            edge_dict["pred_valid"],
        ):
            pair = (int(source), int(target))
            if bool(matched):
                status = "TP"
            elif bool(valid):
                status = "FP"
            else:
                status = "ignored"
            final_edge_status[pair] = status

        mask = audit["video"].astype(str) == video
        indices = audit.index[mask]

        for idx in indices:
            row = audit.loc[idx]
            parent = _safe_int(row.get("parent"))
            existing = _safe_int(row.get("existing_child"))
            candidate = _safe_int(row.get("candidate_child"))

            parent_gt = pred_to_gt.get(parent) if parent is not None else None
            existing_gt = pred_to_gt.get(existing) if existing is not None else None
            candidate_gt = pred_to_gt.get(candidate) if candidate is not None else None

            if parent_gt is not None:
                audit.at[idx, "parent_gt"] = parent_gt
                audit.at[idx, "gt_parent_outdegree"] = int(gt_outdegree.get(parent_gt, 0))
            if existing_gt is not None:
                audit.at[idx, "existing_child_gt"] = existing_gt
            if candidate_gt is not None:
                audit.at[idx, "candidate_child_gt"] = candidate_gt

            existing_is_gt = (
                parent_gt is not None
                and existing_gt is not None
                and (parent_gt, existing_gt) in gt_edges
            )
            candidate_is_gt = (
                parent_gt is not None
                and candidate_gt is not None
                and (parent_gt, candidate_gt) in gt_edges
            )

            audit.at[idx, "existing_edge_is_exact_gt"] = bool(existing_is_gt)
            audit.at[idx, "candidate_edge_is_exact_gt"] = bool(candidate_is_gt)
            audit.at[idx, "proposal_is_exact_gt_division_pair"] = bool(
                gt_outdegree.get(parent_gt, 0) == 2
                and existing_is_gt
                and candidate_is_gt
                and existing_gt != candidate_gt
            )

            # Only label final-edge metric status when that source->candidate edge
            # actually exists in the saved prediction graph.
            if parent is not None and candidate is not None:
                audit.at[idx, "added_edge_metric_status"] = final_edge_status.get(
                    (parent, candidate),
                    "not_added",
                )

        accepted_mask = mask & (audit["decision"].astype(str) == "accepted")
        accepted = audit.loc[accepted_mask]
        status_counts = accepted["added_edge_metric_status"].value_counts().to_dict()
        exact_pair_count = int(
            audit.loc[mask, "proposal_is_exact_gt_division_pair"].sum()
        )

        per_video_summaries.append(
            {
                "video": video,
                "edge_tp": result.edge_tp,
                "edge_fp": result.edge_fp,
                "edge_fn": result.edge_fn,
                "division_tp": result.division_tp,
                "division_fp": result.division_fp,
                "division_fn": result.division_fn,
                "accepted_rescues": int(accepted_mask.sum()),
                "accepted_rescue_edge_TP": int(status_counts.get("TP", 0)),
                "accepted_rescue_edge_FP": int(status_counts.get("FP", 0)),
                "accepted_rescue_edge_ignored": int(status_counts.get("ignored", 0)),
                "exact_true_pair_proposals": exact_pair_count,
            }
        )

        print(
            f"[{video}] accepted={int(accepted_mask.sum())} "
            f"rescue_edge_status={status_counts} "
            f"exact_true_pair_proposals={exact_pair_count} "
            f"div={result.division_tp}/{result.division_fp}/{result.division_fn}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(f"\nSaved labeled audit: {args.output}")

    summary = pd.DataFrame(per_video_summaries)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    if not summary.empty:
        print("\n=== RESCUE EDGE IMPACT ===")
        print(summary.to_string(index=False))

    # Most important diagnostic: locate exact GT division-pair proposals in 5c.
    target_video = "6bba_5c824876"
    target = audit[
        (audit["video"].astype(str) == target_video)
        & (audit["proposal_is_exact_gt_division_pair"])
    ].copy()

    print(f"\n=== EXACT GT DIVISION-PAIR PROPOSALS: {target_video} ===")
    if target.empty:
        print("None found with exact t->t+1 GT-edge matching.")
        print("If V15's TP used the evaluator's ±1 division-time tolerance, inspect nearby GT dividers next.")
    else:
        show = [
            "parent", "frame", "existing_child", "candidate_child",
            "parent_gt", "existing_child_gt", "candidate_child_gt",
            "head_prob", "existing_edge_prob", "candidate_edge_prob",
            "pair_score", "pair_margin", "num_competing_candidates",
            "existing_forward_len", "candidate_forward_len", "parent_history_len",
            "decision", "added_edge_metric_status",
        ]
        show = [c for c in show if c in target.columns]
        print(target[show].sort_values("pair_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
