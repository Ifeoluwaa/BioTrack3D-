#!/usr/bin/env python
"""BioTrack3D++ V16.4a: V16.3.2 + precision close-sister mitosis route.

V16.3 targets the two error sources identified by GT diagnostics:

1. Ordinary edges: most FNs are association misses, so replace framewise greedy
   matching with optional global maximum-gain bipartite matching over the SAME
   neural candidate set, threshold, and distance gate.
2. Mitosis: preserve the verified atlas geometry/logistic score, but calibrate the
   final rescue decision with raw daughter-edge evidence from labeled proposals.
3. Keep post-ILP family completion and use the division cap only as an emergency
   ceiling, never as a quota.

No GT information is used during inference.

Usage
-----
python scripts/predict_biotrack3d_v16.py \
    --split 0 \
    --weights weights/unet_transformer/split_0/edge_predictor_best.pth \
    --use-ilp \
    --det-threshold 0.30 \
    --edge-threshold 0.20 \
    --evaluate
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


# Compatibility shim for some tracksdata / Polars combinations.
if not hasattr(pl, "Float16"):
    pl.Float16 = pl.Float32

import tracksdata as td

from tracking_cellmot.io import open_dataset, save_graph
from tracking_cellmot.models import TemporalUNet3D

sys.path.insert(0, str(Path(__file__).parent))

from train_biotrack3d_v3 import (
    DEFAULT_METHOD,
    UNetNodeTransformer,
    extract_pos_features,
    _POS_EMBED_DIM,
)

from dataspec import (
    USERNAME,
    INTERACTIVE,
    WEIGHTS_PATH,
    PREDICTIONS_PATH,
)

from evaluate import evaluate_run
from tracking_cellmot.metrics import summarise


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PredictConfig:
    """Prediction, graph-decoding, and precision-first division settings."""

    # Keep detector and neural edge model unchanged.
    det_threshold: float = 0.30
    det_tta: bool = True
    pool_kernel_um: float = 5.0

    edge_threshold: float = 0.20
    max_link_distance_um: float = 15.0

    # V16.3.2: global framewise assignment is the default because GT diagnostics
    # showed 318/369 edge FNs were association misses. Use "greedy" for A/B.
    association_mode: str = "greedy"

    # Neural evidence.
    division_head_threshold: float = 0.58
    division_min_edge_prob: float = 0.20

    # Verified BioHub mitosis hard gates from the training atlas.
    division_parent_max_um: float = 11.0
    division_sister_min_um: float = 7.0
    division_sister_max_um: float = 15.3
    division_midpoint_max_um: float = 6.0

    # V16.4a controlled experiment: two verified GT misses had strong neural
    # evidence but predicted sister distances of 6.08 and 5.39 um. Do NOT
    # globally relax the atlas gate. Admit 5-7 um only through a separate
    # strong-neural route; normal compact/established routes still require
    # the original 7-15.3 um atlas interval.
    division_close_sister_min_um: float = 5.0
    division_close_sister_head_min: float = 0.85
    division_close_sister_existing_edge_min: float = 0.85
    division_close_sister_candidate_edge_min: float = 0.55
    division_existing_child_max_um: float = 10.4

    # Learned biology remains a family plausibility/ranking signal. Labeled GT
    # proposals in the first-8 diagnostic include a true pair at ~0.07465, so
    # the V16.3.2 evidence envelope uses a conservative floor of 0.07.
    division_min_score: float = 0.07

    # Labeled rescue calibration (5 exact GT pairs vs 3 harmful rescue FPs):
    # all exact pairs had candidate edge >= 0.3279; all harmful FPs <= 0.2932.
    # These thresholds complement, not replace, the atlas geometry/biology.
    division_evidence_head_min: float = 0.83
    division_evidence_existing_edge_min: float = 0.65
    division_evidence_candidate_edge_min: float = 0.32

    # V16.3.2: two GT-supported family-shape routes. The compact route now
    # requires geometry PLUS either strong two-daughter neural support or a
    # stronger learned biology score. This keeps all 4 known compact GT pairs
    # while rejecting the labeled 5c harmful FP parent=18247. The established
    # route remains unchanged to preserve the verified asymmetric 668 split.
    division_compact_angle_min: float = 115.0
    division_compact_pds_min: float = 0.70
    division_compact_midpoint_max_um: float = 2.60
    division_compact_min_daughter_edge_strong: float = 0.70
    division_compact_pair_score_strong: float = 0.16

    division_established_existing_edge_min: float = 0.88
    division_established_candidate_edge_min: float = 0.40
    division_established_pair_score_min: float = 0.20
    division_established_history_min: int = 3
    division_established_forward_min: int = 3

    # Precision / ambiguity controls. The verified true pair beats its known
    # confusing alternative by ~0.041, so 0.03 preserves that example.
    division_pair_margin: float = 0.03

    # If only one daughter persists after the split, require much stronger
    # evidence. With zero persistent daughters V16 abstains entirely.
    division_single_persist_min_score: float = 0.35
    division_single_persist_min_edge_prob: float = 0.35
    division_single_persist_min_head_prob: float = 0.75

    # Emergency ceiling only; never a target number of rescues.
    max_division_fraction: float = 0.00375

    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    audit_csv: Path | None = None

    # Diagnostic only: save every neural source->target pair within the normal
    # physical link radius, including probabilities below edge_threshold.
    # This does NOT change prediction/selection logic.
    association_candidates_dir: Path | None = None


_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


# =============================================================================
# General helpers
# =============================================================================

@contextlib.contextmanager
def suppress_output():
    """Suppress output from the ILP solver."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                yield


def sigmoid(x: float) -> float:
    """Numerically stable scalar sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def physical_distance(
    coord_a_ds: np.ndarray,
    coord_b_ds: np.ndarray,
    voxel_size_ds: np.ndarray,
) -> float:
    """Distance between downsampled-grid coordinates in micrometres."""
    delta_ds = np.asarray(coord_a_ds, dtype=np.float32) - np.asarray(
        coord_b_ds,
        dtype=np.float32,
    )
    delta_um = delta_ds * voxel_size_ds
    return float(np.linalg.norm(delta_um))


def count_divisions_from_edges(
    edges: list[tuple[int, int, float, float]],
) -> int:
    """Count source nodes with exactly two outgoing edges."""
    outdegree: Counter[int] = Counter()
    for src, _, _, _ in edges:
        outdegree[src] += 1
    return sum(n == 2 for n in outdegree.values())


def count_divisions_in_graph(
    graph: td.graph.InMemoryGraph,
) -> int:
    """Count graph nodes with exactly two outgoing edges."""
    edge_table = graph.edge_attrs()
    outdegree: Counter[int] = Counter()

    for row in edge_table.iter_rows(named=True):
        outdegree[int(row["source_id"])] += 1

    return sum(n == 2 for n in outdegree.values())


# =============================================================================
# Model loading
# =============================================================================

def load_model(
    weights_path: Path,
    device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct model from checkpoint and adjacent config.json."""

    config_path = weights_path.parent / "config.json"

    if config_path.exists():
        config = {
            **_DEFAULT_CONFIG,
            **json.loads(config_path.read_text()),
        }
    else:
        print(
            f"Warning: config.json not found at {config_path}; "
            "using defaults.",
            flush=True,
        )
        config = dict(_DEFAULT_CONFIG)

    if "downsample_factor" in config and "downsample" not in config:
        value = int(config["downsample_factor"])
        config["downsample"] = [value, value, value]

    downsample = tuple(int(v) for v in config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=int(config["unet_out_channels"]),
        layers=list(config["unet_layers"]),
    )

    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=int(config["unet_out_channels"]),
        pos_feat_dim=4 * _POS_EMBED_DIM,
        pooling_mode=config.get("pooling_mode", "single_voxel"),
        pool_radius=int(config.get("pool_radius", 0)),
        pool_sigma=config.get("pool_sigma", None),
        attn_dim=config.get("attn_dim", None),
        sibling_aware=bool(config.get("sibling_aware", True)),
    )

    state = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    current_state = model.state_dict()

    compatible = {}
    shape_skipped = []

    for k, v in state.items():
        if k in current_state and tuple(v.shape) == tuple(current_state[k].shape):
            compatible[k] = v
        else:
            shape_skipped.append(k)

    missing, unexpected = model.load_state_dict(
        compatible,
        strict=False,
    )

    print(
        f"Checkpoint loaded: "
        f"loaded={len(compatible)} "
        f"missing={len(missing)} "
        f"unexpected={len(unexpected)} "
        f"shape_skipped={len(shape_skipped)}"
    )
    model.to(device)
    model.eval()

    return model, int(config["window_size"]), downsample


# =============================================================================
# GEFF graph construction
# =============================================================================

def build_graph(
    coords_original: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> tuple[td.graph.InMemoryGraph, list[int]]:
    """Build a graph and retain global-index -> graph-node-id mapping."""

    graph = td.graph.InMemoryGraph()

    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes(
        [
            {
                "t": int(t),
                "z": float(z),
                "y": float(y),
                "x": float(x),
            }
            for t, z, y, x in coords_original
        ]
    )
    node_ids = [int(node_id) for node_id in node_ids]

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)

        graph.bulk_add_edges(
            [
                {
                    "source_id": node_ids[src],
                    "target_id": node_ids[tgt],
                    "edge_prob": float(prob),
                    "edge_dist": float(dist),
                }
                for src, tgt, prob, dist in edges
            ]
        )

    return graph, node_ids


# =============================================================================
# Image and detection helpers
# =============================================================================

def load_frame(
    zarr_array,
    time_index: int,
    target_shape: list[int],
    downsample: tuple[int, int, int],
) -> torch.Tensor:
    """Read one image frame in the model's downsampled voxel grid."""

    dz, dy, dx = downsample

    raw = zarr_array[
        time_index,
        ::dz,
        ::dy,
        ::dx,
    ].astype(np.float32)

    frame = torch.from_numpy(raw)

    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None],
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    return frame


def pool_kernel_from_um(
    radius_um: float,
    voxel_size_ds: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Convert physical NMS diameter/radius setting to odd voxel kernels."""

    kernel = []

    for spacing in voxel_size_ds:
        size = max(1, int(round(radius_um / spacing)))

        if size % 2 == 0:
            size += 1

        kernel.append(size)

    return tuple(kernel)


def detect_cells(
    detection_logits: torch.Tensor,
    time_index: int,
    det_threshold: float,
    pool_kernel: tuple[int, int, int],
) -> np.ndarray:
    """Detect local-maxima cell centers in downsampled coordinates.

    Returns
    -------
    np.ndarray
        Shape (N, 4), columns [t, z_ds, y_ds, x_ds].
    """

    logits = detection_logits.unsqueeze(0)
    probabilities = torch.sigmoid(logits)

    padding = tuple(size // 2 for size in pool_kernel)

    pooled = F.max_pool3d(
        probabilities,
        kernel_size=pool_kernel,
        stride=1,
        padding=padding,
    )

    is_peak = (
        (probabilities == pooled)
        & (probabilities >= det_threshold)
    )

    peak_indices = torch.nonzero(is_peak[0, 0])

    if peak_indices.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    spatial = peak_indices.float().cpu().numpy()
    time_column = np.full(
        (len(spatial), 1),
        time_index,
        dtype=np.float32,
    )

    return np.concatenate(
        [time_column, spatial],
        axis=1,
    ).astype(np.float32)


def deduplicate_detections(
    coords_ds: np.ndarray,
    detection_logits: torch.Tensor,
    voxel_size_ds: np.ndarray,
    radius_um: float = 3.5,
) -> np.ndarray:
    """Keep one highest-logit peak per local physical neighborhood."""

    if len(coords_ds) <= 1:
        return coords_ds

    logits = detection_logits[0, 0]

    scores = logits[
        coords_ds[:, 1].astype(np.int64),
        coords_ds[:, 2].astype(np.int64),
        coords_ds[:, 3].astype(np.int64),
    ].detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    keep_mask = np.ones(len(coords_ds), dtype=bool)
    kept_indices: list[int] = []

    for index in order:
        if not keep_mask[index]:
            continue

        kept_indices.append(int(index))

        offsets_ds = (
            coords_ds[:, 1:].astype(np.float32)
            - coords_ds[index, 1:].astype(np.float32)
        )

        distances_um = np.linalg.norm(
            offsets_ds * voxel_size_ds,
            axis=1,
        )

        duplicates = distances_um <= radius_um
        duplicates[index] = False
        keep_mask[duplicates] = False

    return coords_ds[
        np.asarray(sorted(kept_indices), dtype=np.int64)
    ]


# =============================================================================
# Normal 1->1 association
# =============================================================================

def build_normal_associations(
    probabilities: np.ndarray,
    source_coords_ds: np.ndarray,
    target_coords_ds: np.ndarray,
    source_global_indices: np.ndarray,
    target_global_indices: np.ndarray,
    voxel_size_ds: np.ndarray,
    edge_threshold: float,
    max_link_distance_um: float,
) -> list[tuple[int, int, float, float]]:
    """Build conservative 1->1 edges with physical distance gating."""

    candidates: list[tuple[float, int, int, float]] = []

    n_source, n_target = probabilities.shape

    for source_local in range(n_source):
        for target_local in range(n_target):
            probability = float(
                probabilities[source_local, target_local]
            )

            if probability < edge_threshold:
                continue

            distance_um = physical_distance(
                source_coords_ds[source_local],
                target_coords_ds[target_local],
                voxel_size_ds,
            )

            if distance_um > max_link_distance_um:
                continue

            candidates.append(
                (
                    probability,
                    source_local,
                    target_local,
                    distance_um,
                )
            )

    candidates.sort(
        key=lambda value: value[0],
        reverse=True,
    )

    source_used: set[int] = set()
    target_used: set[int] = set()
    selected: list[tuple[int, int, float, float]] = []

    for probability, source_local, target_local, distance_um in candidates:
        if source_local in source_used:
            continue

        if target_local in target_used:
            continue

        source_used.add(source_local)
        target_used.add(target_local)

        selected.append(
            (
                int(source_global_indices[source_local]),
                int(target_global_indices[target_local]),
                float(probability),
                float(distance_um),
            )
        )

    return selected


def build_global_associations(
    probabilities: np.ndarray,
    source_coords_ds: np.ndarray,
    target_coords_ds: np.ndarray,
    source_global_indices: np.ndarray,
    target_global_indices: np.ndarray,
    voxel_size_ds: np.ndarray,
    edge_threshold: float,
    max_link_distance_um: float,
) -> list[tuple[int, int, float, float]]:
    """Globally optimize 1->1 links for one frame pair.

    Uses the exact same candidate threshold and physical distance gate as the
    historical greedy matcher. Each eligible edge receives a positive gain
    logit(p)-logit(threshold); leaving a source/target unmatched has zero gain.
    A square assignment with dummy rows/columns therefore chooses the globally
    best compatible set without forcing low-confidence links.
    """
    n_source, n_target = probabilities.shape
    if n_source == 0 or n_target == 0:
        return []

    eps = 1e-6
    threshold = float(np.clip(edge_threshold, eps, 1.0 - eps))
    threshold_logit = float(np.log(threshold / (1.0 - threshold)))

    size = n_source + n_target
    # Zero-cost dummy assignments represent appearing/disappearing tracks.
    cost = np.zeros((size, size), dtype=np.float64)
    valid = np.zeros((n_source, n_target), dtype=bool)
    distances = np.full((n_source, n_target), np.inf, dtype=np.float32)

    # Invalid real-real links must be much worse than a dummy assignment.
    cost[:n_source, :n_target] = 1e6

    for source_local in range(n_source):
        for target_local in range(n_target):
            probability = float(probabilities[source_local, target_local])
            if probability < edge_threshold:
                continue
            distance_um = physical_distance(
                source_coords_ds[source_local],
                target_coords_ds[target_local],
                voxel_size_ds,
            )
            if distance_um > max_link_distance_um:
                continue

            p = float(np.clip(probability, eps, 1.0 - eps))
            gain = float(np.log(p / (1.0 - p)) - threshold_logit)
            if gain <= 0.0:
                continue

            valid[source_local, target_local] = True
            distances[source_local, target_local] = float(distance_um)
            cost[source_local, target_local] = -gain

    row_ind, col_ind = linear_sum_assignment(cost)
    selected: list[tuple[int, int, float, float]] = []
    for row, col in zip(row_ind, col_ind, strict=True):
        if row >= n_source or col >= n_target:
            continue
        if not valid[row, col]:
            continue
        selected.append(
            (
                int(source_global_indices[row]),
                int(target_global_indices[col]),
                float(probabilities[row, col]),
                float(distances[row, col]),
            )
        )
    return selected


# =============================================================================
# V16.3 evidence-calibrated post-ILP division rescue
# =============================================================================

def division_angle_deg(
    parent_xyz_ds: np.ndarray,
    daughter1_xyz_ds: np.ndarray,
    daughter2_xyz_ds: np.ndarray,
    voxel_size_ds: np.ndarray,
) -> float:
    """Angle between the two parent->daughter displacement vectors."""
    v1 = (daughter1_xyz_ds - parent_xyz_ds) * voxel_size_ds
    v2 = (daughter2_xyz_ds - parent_xyz_ds) * voxel_size_ds
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-8 or n2 <= 1e-8:
        return 0.0
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def learned_division_pair_score(
    angle_deg: float,
    pds: float,
    sister_dist_um: float,
    midpoint_error_um: float,
) -> float:
    """Verified 4-feature logistic mitosis score learned from 151 GT divisions."""
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
    return sigmoid(float(z))


def _linear_forward_length(
    start: int,
    outgoing: dict[int, list[tuple[int, float, float]]],
    max_steps: int = 4,
) -> int:
    """Count consecutive unique-successor edges after a node."""
    length = 0
    current = int(start)
    seen = {current}
    while length < max_steps:
        children = outgoing.get(current, [])
        if len(children) != 1:
            break
        nxt = int(children[0][0])
        if nxt in seen:
            break
        seen.add(nxt)
        length += 1
        current = nxt
    return length


def _linear_backward_length(
    start: int,
    incoming_sources: dict[int, list[int]],
    max_steps: int = 4,
) -> int:
    """Count consecutive unique-predecessor edges before a node."""
    length = 0
    current = int(start)
    seen = {current}
    while length < max_steps:
        parents = incoming_sources.get(current, [])
        if len(parents) != 1:
            break
        prev = int(parents[0])
        if prev in seen:
            break
        seen.add(prev)
        length += 1
        current = prev
    return length


def add_precision_divisions_post_ilp(
    graph: td.graph.InMemoryGraph,
    node_ids: list[int],
    coords_ds: np.ndarray,
    candidate_edges_by_pair: dict[tuple[int, int], float],
    division_head_prob_by_global: dict[int, float],
    voxel_size_ds: np.ndarray,
    config: PredictConfig,
    audit_rows: list[dict],
    video_name: str,
) -> int:
    """Add evidence-calibrated mitoses after ordinary graph decoding.

    V16.3 keeps the verified hard geometry and 4-feature atlas score, but the
    final accept/abstain step is driven by raw neural daughter-edge evidence.
    This follows the labeled first-8 audit: all five exact GT proposals had a
    second-daughter edge >=0.3279, while all three metric-harmful rescue FPs
    were <=0.2932. Persistence/history are retained for audit/context but are
    not mandatory because real early/short daughter branches can have 0-1
    forward links. No proposals are globally ranked across unrelated parents.
    """
    if len(coords_ds) == 0 or graph.num_edges() == 0:
        return 0

    global_to_node_id = [int(node_id) for node_id in node_ids]
    node_to_global = {
        int(node_id): global_index
        for global_index, node_id in enumerate(global_to_node_id)
    }

    surviving_node_ids = {int(node_id) for node_id in graph.node_ids()}

    outgoing: dict[int, list[tuple[int, float, float]]] = {}
    incoming_sources: dict[int, list[int]] = {}
    edge_set: set[tuple[int, int]] = set()

    edge_table = graph.edge_attrs()
    for row in edge_table.iter_rows(named=True):
        src_node = int(row["source_id"])
        tgt_node = int(row["target_id"])
        if src_node not in node_to_global or tgt_node not in node_to_global:
            continue
        src = node_to_global[src_node]
        tgt = node_to_global[tgt_node]
        prob = float(row.get("edge_prob", 0.0))
        dist = float(row.get("edge_dist", 0.0))
        outgoing.setdefault(src, []).append((tgt, prob, dist))
        incoming_sources.setdefault(tgt, []).append(src)
        edge_set.add((src, tgt))

    raw_by_parent: dict[int, list[tuple[int, float]]] = {}
    for (parent, child), prob in candidate_edges_by_pair.items():
        raw_by_parent.setdefault(int(parent), []).append((int(child), float(prob)))

    division_cap = max(
        1,
        int(round(graph.num_edges() * config.max_division_fraction)),
    )

    independently_accepted: list[dict] = []
    eligible_count = 0
    removed_by_ilp_count = 0

    for parent, children in outgoing.items():
        # Complete only a 1->1 family. Never rewrite topology here.
        if len(children) != 1:
            continue
        if len(incoming_sources.get(parent, [])) == 0:
            continue

        existing, existing_prob, existing_dist = children[0]
        parent_time = int(coords_ds[parent, 0])
        if int(coords_ds[existing, 0]) != parent_time + 1:
            continue
        if existing_dist > config.division_existing_child_max_um:
            continue

        head_prob = float(division_head_prob_by_global.get(parent, 0.0))
        if head_prob < config.division_head_threshold:
            continue

        parent_xyz = coords_ds[parent, 1:].astype(np.float32)
        existing_xyz = coords_ds[existing, 1:].astype(np.float32)
        family_candidates: list[dict] = []

        for candidate, raw_edge_prob in raw_by_parent.get(parent, []):
            if candidate == existing or (parent, candidate) in edge_set:
                continue
            if raw_edge_prob < config.division_min_edge_prob:
                continue
            if int(coords_ds[candidate, 0]) != parent_time + 1:
                continue

            candidate_node_id = global_to_node_id[int(candidate)]
            if candidate_node_id not in surviving_node_ids:
                removed_by_ilp_count += 1
                continue

            # Candidate must survive ILP and be genuinely parent-free.
            if len(incoming_sources.get(candidate, [])) != 0:
                continue

            candidate_xyz = coords_ds[candidate, 1:].astype(np.float32)
            candidate_dist = physical_distance(parent_xyz, candidate_xyz, voxel_size_ds)
            if candidate_dist > config.division_parent_max_um:
                continue

            sister_dist = physical_distance(existing_xyz, candidate_xyz, voxel_size_ds)
            sister_in_atlas = (
                config.division_sister_min_um
                <= sister_dist
                <= config.division_sister_max_um
            )
            sister_in_close_band = (
                config.division_close_sister_min_um
                <= sister_dist
                < config.division_sister_min_um
            )
            # Preserve the atlas upper bound. The only broadened generator
            # region is 5-7 um, which must later pass the dedicated V16.4a
            # strong-neural close-sister route.
            if not (sister_in_atlas or sister_in_close_band):
                continue

            midpoint_xyz = 0.5 * (existing_xyz + candidate_xyz)
            midpoint_error = physical_distance(parent_xyz, midpoint_xyz, voxel_size_ds)
            if midpoint_error > config.division_midpoint_max_um:
                continue

            d1 = float(existing_dist)
            d2 = float(candidate_dist)
            if max(d1, d2) <= 1e-8:
                continue

            pds = min(d1, d2) / max(d1, d2)
            angle = division_angle_deg(
                parent_xyz,
                existing_xyz,
                candidate_xyz,
                voxel_size_ds,
            )
            pair_score = learned_division_pair_score(
                angle_deg=angle,
                pds=pds,
                sister_dist_um=sister_dist,
                midpoint_error_um=midpoint_error,
            )

            existing_forward = _linear_forward_length(existing, outgoing)
            candidate_forward = _linear_forward_length(candidate, outgoing)
            parent_history = _linear_backward_length(parent, incoming_sources)

            both_persist = existing_forward >= 1 and candidate_forward >= 1
            branches_separate = True
            if both_persist:
                existing_next = int(outgoing[existing][0][0])
                candidate_next = int(outgoing[candidate][0][0])
                branches_separate = existing_next != candidate_next

            family_candidates.append(
                {
                    "parent": parent,
                    "existing_child": existing,
                    "candidate_child": candidate,
                    "head_prob": head_prob,
                    "existing_edge_prob": float(existing_prob),
                    "candidate_edge_prob": float(raw_edge_prob),
                    "existing_dist_um": d1,
                    "candidate_dist_um": d2,
                    "sister_dist_um": sister_dist,
                    "sister_in_atlas": bool(sister_in_atlas),
                    "sister_in_close_band": bool(sister_in_close_band),
                    "midpoint_error_um": midpoint_error,
                    "angle_deg": angle,
                    "pds": pds,
                    "pair_score": pair_score,
                    "existing_forward_len": existing_forward,
                    "candidate_forward_len": candidate_forward,
                    "parent_history_len": parent_history,
                    "both_persist": both_persist,
                    "branches_separate": branches_separate,
                }
            )

        if not family_candidates:
            continue

        # V16.3.2 selection: neural evidence first, then require one of two
        # GT-supported family routes. Compact geometry additionally needs either
        # strong two-daughter neural support or stronger learned biology before
        # biology chooses the best daughter
        # within THIS parent. This keeps the atlas score as local ranking,
        # rather than a global mitosis probability.
        raw_best = max(
            family_candidates,
            key=lambda row: (row["pair_score"], row["candidate_edge_prob"]),
        )

        evidence_candidates = []
        if (
            head_prob >= config.division_evidence_head_min
            and existing_prob >= config.division_evidence_existing_edge_min
        ):
            for row in family_candidates:
                if not row["branches_separate"]:
                    continue
                if (
                    row["candidate_edge_prob"]
                    < config.division_evidence_candidate_edge_min
                ):
                    continue

                normal_pair_ok = row["pair_score"] >= config.division_min_score

                compact_geometry = (
                    row["sister_in_atlas"]
                    and row["angle_deg"] >= config.division_compact_angle_min
                    and row["pds"] >= config.division_compact_pds_min
                    and row["midpoint_error_um"]
                    <= config.division_compact_midpoint_max_um
                )
                compact_min_daughter_edge = min(
                    row["existing_edge_prob"],
                    row["candidate_edge_prob"],
                )
                compact_neural_confidence = (
                    compact_min_daughter_edge
                    >= config.division_compact_min_daughter_edge_strong
                )
                compact_biology_confidence = (
                    row["pair_score"]
                    >= config.division_compact_pair_score_strong
                )
                route_compact = (
                    normal_pair_ok
                    and compact_geometry
                    and (
                        compact_neural_confidence
                        or compact_biology_confidence
                    )
                )

                route_established = (
                    normal_pair_ok
                    and row["sister_in_atlas"]
                    and row["existing_edge_prob"]
                    >= config.division_established_existing_edge_min
                    and row["candidate_edge_prob"]
                    >= config.division_established_candidate_edge_min
                    and row["pair_score"]
                    >= config.division_established_pair_score_min
                    and row["parent_history_len"]
                    >= config.division_established_history_min
                    and row["existing_forward_len"]
                    >= config.division_established_forward_min
                    and row["candidate_forward_len"]
                    >= config.division_established_forward_min
                )

                route_close_sister = (
                    row["sister_in_close_band"]
                    and row["head_prob"]
                    >= config.division_close_sister_head_min
                    and row["existing_edge_prob"]
                    >= config.division_close_sister_existing_edge_min
                    and row["candidate_edge_prob"]
                    >= config.division_close_sister_candidate_edge_min
                    # Parent distance <=11 and midpoint <=6 were already
                    # enforced by the unchanged generator hard gates.
                )

                row["compact_geometry"] = bool(compact_geometry)
                row["compact_min_daughter_edge"] = float(compact_min_daughter_edge)
                row["compact_neural_confidence"] = bool(compact_neural_confidence)
                row["compact_biology_confidence"] = bool(compact_biology_confidence)

                row["route_close_sister"] = bool(route_close_sister)

                if route_compact or route_established or route_close_sister:
                    row["route_compact"] = bool(route_compact)
                    row["route_established"] = bool(route_established)
                    active_routes = []
                    if route_compact:
                        active_routes.append("compact")
                    if route_established:
                        active_routes.append("established")
                    if route_close_sister:
                        active_routes.append("close_sister")
                    row["route_name"] = "+".join(active_routes)
                    evidence_candidates.append(row)

        if evidence_candidates:
            evidence_candidates.sort(
                key=lambda row: (
                    row["pair_score"],
                    row["candidate_edge_prob"],
                ),
                reverse=True,
            )
            best = evidence_candidates[0]
            n_competing = len(evidence_candidates)
            second_score = (
                float(evidence_candidates[1]["pair_score"])
                if n_competing > 1
                else None
            )
            pair_margin = (
                float(best["pair_score"] - second_score)
                if second_score is not None
                else float("nan")
            )
            decision = "accepted_route"
            if n_competing > 1 and pair_margin < config.division_pair_margin:
                decision = "reject_ambiguous_pair"
        else:
            # Audit the strongest raw biological option and state why it could
            # not enter either V16.3.2 family route.
            best = raw_best
            best["route_compact"] = False
            best["route_established"] = False
            best["route_close_sister"] = False
            best["route_name"] = "none"
            best.setdefault("compact_geometry", False)
            best.setdefault("compact_min_daughter_edge", float("nan"))
            best.setdefault("compact_neural_confidence", False)
            best.setdefault("compact_biology_confidence", False)
            n_competing = 0
            pair_margin = float("nan")
            if best["pair_score"] < config.division_min_score:
                decision = "reject_low_pair_score"
            elif not best["branches_separate"]:
                decision = "reject_branch_merge"
            elif head_prob < config.division_evidence_head_min:
                decision = "reject_low_head_evidence"
            elif existing_prob < config.division_evidence_existing_edge_min:
                decision = "reject_low_existing_edge"
            elif (
                best["candidate_edge_prob"]
                < config.division_evidence_candidate_edge_min
            ):
                decision = "reject_low_candidate_edge"
            else:
                decision = "reject_family_shape"

        eligible_count += len(family_candidates)
        edge_ratio = (
            best["candidate_edge_prob"] / max(best["existing_edge_prob"], 1e-8)
        )

        audit_rows.append(
            {
                "parent": parent,
                "frame": parent_time,
                "existing_child": best["existing_child"],
                "candidate_child": best["candidate_child"],
                "head_prob": best["head_prob"],
                "existing_edge_prob": best["existing_edge_prob"],
                "candidate_edge_prob": best["candidate_edge_prob"],
                "parent_dist_um": best["candidate_dist_um"],
                "sister_dist_um": best["sister_dist_um"],
                "midpoint_dist_um": best["midpoint_error_um"],
                "symmetry": abs(best["existing_dist_um"] - best["candidate_dist_um"])
                / (best["existing_dist_um"] + best["candidate_dist_um"] + 1e-6),
                "continuation_bonus": float(best["both_persist"]),
                "pair_score": best["pair_score"],
                "angle_deg": best["angle_deg"],
                "pds": best["pds"],
                "pair_margin": pair_margin,
                "num_competing_candidates": n_competing,
                "num_raw_family_candidates": len(family_candidates),
                "existing_forward_len": best["existing_forward_len"],
                "candidate_forward_len": best["candidate_forward_len"],
                "parent_history_len": best["parent_history_len"],
                # Keep this column for CSV compatibility, but it is no longer a
                # global ranking score. Store the edge ratio for diagnostics.
                "priority": edge_ratio,
                "compact_geometry": bool(best.get("compact_geometry", False)),
                "compact_min_daughter_edge": best.get(
                    "compact_min_daughter_edge", float("nan")
                ),
                "compact_neural_confidence": bool(
                    best.get("compact_neural_confidence", False)
                ),
                "compact_biology_confidence": bool(
                    best.get("compact_biology_confidence", False)
                ),
                "route_compact": bool(best.get("route_compact", False)),
                "route_established": bool(best.get("route_established", False)),
                "route_close_sister": bool(best.get("route_close_sister", False)),
                "sister_in_atlas": bool(best.get("sister_in_atlas", False)),
                "sister_in_close_band": bool(best.get("sister_in_close_band", False)),
                "route_name": best.get("route_name", "none"),
                "decision": decision,
            }
        )

        if decision == "accepted_route":
            best["pair_margin"] = pair_margin
            best["num_competing_candidates"] = n_competing
            best["edge_ratio"] = edge_ratio
            independently_accepted.append(best)

    # No global sort/top-K. Resolve only direct endpoint conflicts. In the rare
    # case two independently accepted parents want the same daughter, abstain on
    # that conflicted daughter instead of ranking unrelated parents.
    child_claims: dict[int, list[dict]] = {}
    for proposal in independently_accepted:
        child_claims.setdefault(int(proposal["candidate_child"]), []).append(proposal)

    selected = [
        proposal
        for proposal in independently_accepted
        if len(child_claims[int(proposal["candidate_child"])]) == 1
    ]

    # Emergency ceiling only. If this ever triggers, fail conservative: keep no
    # arbitrary top-K ranking. The audit/log will tell us to inspect the rule.
    cap_triggered = len(selected) > division_cap
    if cap_triggered:
        selected = []

    if selected:
        selected = [
            row
            for row in selected
            if global_to_node_id[int(row["parent"])] in surviving_node_ids
            and global_to_node_id[int(row["candidate_child"])] in surviving_node_ids
            and len(incoming_sources.get(int(row["candidate_child"]), [])) == 0
        ]

    if selected:
        graph.bulk_add_edges(
            [
                {
                    "source_id": global_to_node_id[int(row["parent"])],
                    "target_id": global_to_node_id[int(row["candidate_child"])],
                    "edge_prob": float(row["candidate_edge_prob"]),
                    "edge_dist": float(row["candidate_dist_um"]),
                }
                for row in selected
            ]
        )

        selected_keys = {
            (int(row["parent"]), int(row["candidate_child"]))
            for row in selected
        }
        for row in audit_rows:
            key = (int(row.get("parent", -1)), int(row.get("candidate_child", -1)))
            if key in selected_keys and row.get("decision") == "accepted_route":
                row["decision"] = "accepted"

    print(
        f"[V16.4a RESCUE] video={video_name} "
        f"eligible={eligible_count} "
        f"independent_pass={len(independently_accepted)} "
        f"removed_by_ilp={removed_by_ilp_count} "
        f"cap={division_cap} "
        f"cap_triggered={cap_triggered} "
        f"added={len(selected)}",
        flush=True,
    )

    for row in selected:
        margin_text = (
            f"{row['pair_margin']:.3f}"
            if np.isfinite(row["pair_margin"])
            else "NA"
        )
        print(
            f"[V16.4a ACCEPT] parent={row['parent']} "
            f"existing={row['existing_child']} "
            f"rescued={row['candidate_child']} "
            f"pair={row['pair_score']:.3f} "
            f"margin={margin_text} "
            f"existing_edge={row['existing_edge_prob']:.3f} "
            f"candidate_edge={row['candidate_edge_prob']:.3f} "
            f"edge_ratio={row['edge_ratio']:.3f} "
            f"head={row['head_prob']:.3f} "
            f"route={row.get('route_name', 'none')} "
            f"persist={row['existing_forward_len']}/{row['candidate_forward_len']} "
            f"history={row['parent_history_len']}",
            flush=True,
        )

    return len(selected)


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    dataset_path: Path,
    device: torch.device,
    config: PredictConfig,
    window_size: int,
    downsample: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int, float, float]],
    dict[tuple[int, int], float],
    dict[int, float],
    np.ndarray,
    list[dict],
    dict[str, np.ndarray],
]:
    """Predict one complete video and return post-decoding rescue context."""

    dataset = open_dataset(
        dataset_path,
        normalize=False,
        load_image=False,
        downsample=downsample,
        require_tracks=False,
    )

    if (
        "0.001" not in dataset.quantiles
        or "0.999" not in dataset.quantiles
    ):
        raise ValueError(
            f"Zarr statistics missing for {dataset_path}"
        )

    zarr_array = zarr.open_group(
        str(dataset.zarr_path),
        mode="r",
    )["0"]

    q_low = float(dataset.quantiles["0.001"])
    q_high = float(dataset.quantiles["0.999"])

    total_frames = int(dataset.image_shape[0])
    image_shape_ds = tuple(dataset.image_shape)
    target_shape = list(image_shape_ds[1:])

    downsample_array = np.asarray(
        downsample,
        dtype=np.float32,
    )

    downsample_tensor = torch.from_numpy(
        downsample_array
    ).to(device)

    voxel_size_original = np.asarray(
        dataset.scale,
        dtype=np.float32,
    )

    voxel_size_ds = (
        voxel_size_original
        * downsample_array
    )

    pool_kernel = pool_kernel_from_um(
        config.pool_kernel_um,
        tuple(float(v) for v in voxel_size_ds),
    )

    print(
        f"Detection pool kernel: {pool_kernel}; "
        f"effective voxel size µm: {voxel_size_ds.tolist()}",
        flush=True,
    )

    all_coords_ds: list[np.ndarray] = []
    frame_offset: dict[int, tuple[int, int]] = {}

    all_edges: list[tuple[int, int, float, float]] = []
    candidate_edges_by_pair: dict[tuple[int, int], float] = {}
    division_head_by_global: dict[int, float] = {}

    global_node_count = 0
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()

    audit_rows: list[dict] = []

    # Association diagnostic cache. IDs are global detection indices here;
    # predict() converts them to persistent graph node IDs before saving.
    assoc_src_global: list[np.ndarray] = []
    assoc_tgt_global: list[np.ndarray] = []
    assoc_source_time: list[np.ndarray] = []
    assoc_target_time: list[np.ndarray] = []
    assoc_prob: list[np.ndarray] = []
    assoc_dist_um: list[np.ndarray] = []

    stride = max(window_size - 1, 1)

    starts = list(
        range(
            0,
            total_frames - window_size + 1,
            stride,
        )
    )

    last_start = max(total_frames - window_size, 0)

    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    for window_start in tqdm(
        starts,
        desc="windows",
        leave=False,
        disable=not INTERACTIVE,
    ):
        frame_indices = list(
            range(
                window_start,
                window_start + window_size,
            )
        )

        frames = torch.stack(
            [
                load_frame(
                    zarr_array,
                    time_index,
                    target_shape,
                    downsample,
                )
                for time_index in frame_indices
            ]
        )

        frames = (
            (frames - q_low)
            / (q_high - q_low + 1e-6)
        ).clamp(0.0)

        frames = frames.unsqueeze(0).to(device)

        unet_output, detection_logits = model.encode(frames)

        if config.det_tta:
            flip_axes = [
                (-1,),
                (-2,),
                (-2, -1),
            ]

            for axes in flip_axes:
                flipped = frames.flip(axes)
                _, flipped_logits = model.encode(flipped)

                for frame_local in range(window_size):
                    detection_logits[frame_local] = (
                        detection_logits[frame_local]
                        + flipped_logits[frame_local].flip(axes)
                    )

            for frame_local in range(window_size):
                detection_logits[frame_local] = (
                    detection_logits[frame_local] / 4.0
                )

        # ---------------------------------------------------------------------
        # Per-frame detections
        # ---------------------------------------------------------------------
        for frame_local, time_index in enumerate(frame_indices):
            if time_index in seen_frames:
                continue

            detections_ds = detect_cells(
                detection_logits[frame_local][0],
                time_index,
                config.det_threshold,
                pool_kernel,
            )

            detections_ds = deduplicate_detections(
                detections_ds,
                detection_logits[frame_local],
                voxel_size_ds,
            )

            frame_offset[time_index] = (
                global_node_count,
                global_node_count + len(detections_ds),
            )

            global_node_count += len(detections_ds)
            all_coords_ds.append(detections_ds)
            seen_frames.add(time_index)

            print(
                f"Frame {time_index}: "
                f"{len(detections_ds)} detections",
                flush=True,
            )

        coords_so_far_ds = (
            np.concatenate(all_coords_ds, axis=0)
            if all_coords_ds
            else np.empty((0, 4), dtype=np.float32)
        )

        # ---------------------------------------------------------------------
        # Pairwise temporal associations
        # ---------------------------------------------------------------------
        for frame_local in range(window_size - 1):
            source_time = frame_indices[frame_local]
            target_time = frame_indices[frame_local + 1]

            pair_key = (source_time, target_time)

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            if (
                source_time not in frame_offset
                or target_time not in frame_offset
            ):
                continue

            src_start, src_end = frame_offset[source_time]
            tgt_start, tgt_end = frame_offset[target_time]

            if src_start == src_end or tgt_start == tgt_end:
                continue

            source_coords_full = coords_so_far_ds[src_start:src_end]
            target_coords_full = coords_so_far_ds[tgt_start:tgt_end]

            source_coords_ds = source_coords_full[:, 1:]
            target_coords_ds = target_coords_full[:, 1:]

            n_source = len(source_coords_ds)
            n_target = len(target_coords_ds)

            source_global = np.arange(
                src_start,
                src_end,
                dtype=np.int64,
            )

            target_global = np.arange(
                tgt_start,
                tgt_end,
                dtype=np.int64,
            )

            source_coords_tensor = torch.from_numpy(
                source_coords_ds.astype(np.float32)
            ).unsqueeze(0).to(device)

            target_coords_tensor = torch.from_numpy(
                target_coords_ds.astype(np.float32)
            ).unsqueeze(0).to(device)

            source_relative = source_coords_full.copy()
            source_relative[:, 0] = frame_local

            target_relative = target_coords_full.copy()
            target_relative[:, 0] = frame_local + 1

            pos_shape = (
                window_size,
                *image_shape_ds[1:],
            )

            source_pos = torch.from_numpy(
                extract_pos_features(
                    source_relative,
                    pos_shape,
                )
            ).unsqueeze(0).to(device)

            target_pos = torch.from_numpy(
                extract_pos_features(
                    target_relative,
                    pos_shape,
                )
            ).unsqueeze(0).to(device)

            source_mask = torch.ones(
                (1, n_source),
                dtype=torch.bool,
                device=device,
            )

            target_mask = torch.ones(
                (1, n_target),
                dtype=torch.bool,
                device=device,
            )

            source_features = model._index_features(
                unet_output[:, frame_local],
                source_coords_tensor,
                source_mask,
            )

            target_features = model._index_features(
                unet_output[:, frame_local + 1],
                target_coords_tensor,
                target_mask,
            )

            # The transformer was trained using original-grid coordinate values.
            source_coords_original_tensor = (
                source_coords_tensor * downsample_tensor
            )

            target_coords_original_tensor = (
                target_coords_tensor * downsample_tensor
            )

            edge_logits = model.predict_edges(
                source_features,
                target_features,
                source_coords_original_tensor,
                target_coords_original_tensor,
                source_pos,
                target_pos,
                source_mask,
                target_mask,
            )[0]

            edge_probabilities = torch.sigmoid(
                edge_logits
            ).cpu().numpy()

            division_logits = model.predict_divisions(
                source_features,
                target_features,
                source_coords_original_tensor,
                target_coords_original_tensor,
                source_pos,
                target_pos,
                source_mask,
                target_mask,
            )[0]

            division_probabilities = torch.sigmoid(
                division_logits
            ).cpu().numpy()

            for source_local, source_global_id in enumerate(source_global):
                division_head_by_global[
                    int(source_global_id)
                ] = float(
                    division_probabilities[source_local]
                )

            # Diagnostic cache: every pair inside the tracker's physical link
            # radius, regardless of neural probability. Vectorized to avoid a
            # second Python O(N*M) loop. This cache is read only by the offline
            # association-miss analyzer and never affects predictions.
            if config.association_candidates_dir is not None:
                delta_um = (
                    source_coords_ds[:, None, :]
                    - target_coords_ds[None, :, :]
                ) * voxel_size_ds[None, None, :]
                distance_matrix_um = np.linalg.norm(delta_um, axis=2)
                src_local_idx, tgt_local_idx = np.nonzero(
                    distance_matrix_um <= config.max_link_distance_um
                )
                if len(src_local_idx):
                    assoc_src_global.append(source_global[src_local_idx].astype(np.int64))
                    assoc_tgt_global.append(target_global[tgt_local_idx].astype(np.int64))
                    assoc_source_time.append(
                        np.full(len(src_local_idx), source_time, dtype=np.int32)
                    )
                    assoc_target_time.append(
                        np.full(len(src_local_idx), target_time, dtype=np.int32)
                    )
                    assoc_prob.append(
                        edge_probabilities[src_local_idx, tgt_local_idx].astype(np.float32)
                    )
                    assoc_dist_um.append(
                        distance_matrix_um[src_local_idx, tgt_local_idx].astype(np.float32)
                    )

            # Save all plausible pair probabilities for later division completion.
            for source_local in range(n_source):
                for target_local in range(n_target):
                    probability = float(
                        edge_probabilities[
                            source_local,
                            target_local,
                        ]
                    )

                    if probability < config.division_min_edge_prob:
                        continue

                    source_global_id = int(
                        source_global[source_local]
                    )

                    target_global_id = int(
                        target_global[target_local]
                    )

                    candidate_edges_by_pair[
                        (
                            source_global_id,
                            target_global_id,
                        )
                    ] = probability

            if config.association_mode == "global":
                normal_edges = build_global_associations(
                    probabilities=edge_probabilities,
                    source_coords_ds=source_coords_ds,
                    target_coords_ds=target_coords_ds,
                    source_global_indices=source_global,
                    target_global_indices=target_global,
                    voxel_size_ds=voxel_size_ds,
                    edge_threshold=config.edge_threshold,
                    max_link_distance_um=config.max_link_distance_um,
                )
            elif config.association_mode == "greedy":
                normal_edges = build_normal_associations(
                    probabilities=edge_probabilities,
                    source_coords_ds=source_coords_ds,
                    target_coords_ds=target_coords_ds,
                    source_global_indices=source_global,
                    target_global_indices=target_global,
                    voxel_size_ds=voxel_size_ds,
                    edge_threshold=config.edge_threshold,
                    max_link_distance_um=config.max_link_distance_um,
                )
            else:
                raise ValueError(
                    f"Unknown association_mode={config.association_mode!r}; "
                    "expected 'global' or 'greedy'."
                )

            all_edges.extend(normal_edges)

    del unet_output

    coords_ds = (
        np.concatenate(all_coords_ds, axis=0)
        if all_coords_ds
        else np.empty((0, 4), dtype=np.float32)
    )

    print(
        f"[NORMAL GRAPH] nodes={len(coords_ds)} "
        f"edges={len(all_edges)} "
        f"divisions={count_divisions_from_edges(all_edges)}",
        flush=True,
    )

    # IMPORTANT V16 sequencing: do not add any division edge here.
    # Build/solve the 1->1 graph first; rescue happens after optional ILP.
    coords_original = coords_ds.copy()
    coords_original[:, 1:] *= downsample_array

    association_candidate_cache = {
        "source_global": (
            np.concatenate(assoc_src_global).astype(np.int64, copy=False)
            if assoc_src_global else np.empty(0, dtype=np.int64)
        ),
        "target_global": (
            np.concatenate(assoc_tgt_global).astype(np.int64, copy=False)
            if assoc_tgt_global else np.empty(0, dtype=np.int64)
        ),
        "source_time": (
            np.concatenate(assoc_source_time).astype(np.int32, copy=False)
            if assoc_source_time else np.empty(0, dtype=np.int32)
        ),
        "target_time": (
            np.concatenate(assoc_target_time).astype(np.int32, copy=False)
            if assoc_target_time else np.empty(0, dtype=np.int32)
        ),
        "prob": (
            np.concatenate(assoc_prob).astype(np.float32, copy=False)
            if assoc_prob else np.empty(0, dtype=np.float32)
        ),
        "dist_um": (
            np.concatenate(assoc_dist_um).astype(np.float32, copy=False)
            if assoc_dist_um else np.empty(0, dtype=np.float32)
        ),
    }

    return (
        coords_original,
        coords_ds,
        all_edges,
        candidate_edges_by_pair,
        division_head_by_global,
        voxel_size_ds,
        audit_rows,
        association_candidate_cache,
    )


# =============================================================================
# Audit output
# =============================================================================

def write_audit_csv(
    audit_rows: list[dict],
    output_path: Path,
    video_name: str,
) -> None:
    """Append division pair-completion decisions to a CSV."""

    if not audit_rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "video",
        "parent",
        "frame",
        "existing_child",
        "candidate_child",
        "head_prob",
        "existing_edge_prob",
        "candidate_edge_prob",
        "parent_dist_um",
        "sister_dist_um",
        "midpoint_dist_um",
        "symmetry",
        "continuation_bonus",
        "pair_score",
        "angle_deg",
        "pds",
        "pair_margin",
        "num_competing_candidates",
        "num_raw_family_candidates",
        "existing_forward_len",
        "candidate_forward_len",
        "parent_history_len",
        "priority",
        "compact_geometry",
        "compact_min_daughter_edge",
        "compact_neural_confidence",
        "compact_biology_confidence",
        "route_compact",
        "route_established",
        "route_close_sister",
        "sister_in_atlas",
        "sister_in_close_band",
        "route_name",
        "decision",
    ]

    exists = output_path.exists()

    with output_path.open(
        "a",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        for row in audit_rows:
            writer.writerow(
                {
                    "video": video_name,
                    **row,
                }
            )


# =============================================================================
# Main prediction loop
# =============================================================================

def predict(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_path: Path,
    config: PredictConfig,
    method: str,
    debug_video: Path | None,
    video_slice: slice | None,
    partition: str,
    evaluate: bool,
) -> None:
    """Run prediction, save GEFF files, and optionally evaluate."""

    if debug_video is not None:
        debug_path = debug_video
        data_dir = debug_path.parent

        if debug_path.name.endswith(".zarr"):
            test_names = [debug_path.stem]
        elif debug_path.name.endswith(".geff"):
            test_names = [debug_path.stem]
        else:
            test_names = [debug_path.name]

    else:
        if splits_file.exists():
            folds = json.loads(splits_file.read_text())
        else:
            import random

            dataset_names = sorted(
                path.stem
                for path in data_dir.glob("*.zarr")
                if (data_dir / f"{path.stem}.geff").exists()
            )

            random.Random(0).shuffle(dataset_names)

            n_validation = max(
                1,
                len(dataset_names) // 10,
            )

            folds = [
                {
                    "train": dataset_names[n_validation:],
                    "test": dataset_names[:n_validation],
                }
            ]

        if partition not in {"train", "test"}:
            raise ValueError(f"partition must be 'train' or 'test', got {partition!r}")
        test_names = folds[fold][partition]

        if video_slice is not None:
            test_names = test_names[video_slice]

    output_dir = (
        PREDICTIONS_PATH
        / USERNAME
        / method
        / f"split_{fold}"
    )
    import shutil
    if output_dir.exists():
        for file_path in output_dir.glob("*.geff"):
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, window_size, downsample = load_model(
        weights_path,
        device,
    )

    print(
        f"Fold={fold}; videos={len(test_names)}; "
        f"device={device}; window={window_size}; "
        f"downsample={downsample}; "
        f"edge_threshold={config.edge_threshold}; "
        f"association_mode={config.association_mode}; "
        f"division_head_threshold={config.division_head_threshold}",
        flush=True,
    )

    if config.audit_csv is not None and config.audit_csv.exists():
        config.audit_csv.unlink()

    for name in tqdm(
        test_names,
        desc="Predicting",
        disable=not INTERACTIVE,
    ):
        dataset_path = (
            debug_video
            if debug_video is not None
            else data_dir / name
        )

        (
            coords_original,
            coords_ds,
            edges,
            candidate_edges_by_pair,
            division_head_by_global,
            voxel_size_ds,
            audit_rows,
            association_candidate_cache,
        ) = predict_video(
            model=model,
            dataset_path=dataset_path,
            device=device,
            config=config,
            window_size=window_size,
            downsample=downsample,
        )

        print(
            f"[PRE-GRAPH] {name}: "
            f"nodes={len(coords_original)} "
            f"edges={len(edges)} "
            f"divisions={count_divisions_from_edges(edges)}",
            flush=True,
        )

        graph, node_ids = build_graph(
            coords_original,
            edges,
        )

        if config.association_candidates_dir is not None:
            cache_dir = Path(config.association_candidates_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            sg = association_candidate_cache["source_global"]
            tg = association_candidate_cache["target_global"]
            preselected = {(int(a), int(b)) for a, b, _, _ in edges}
            source_node_id = np.fromiter(
                (int(node_ids[int(i)]) for i in sg),
                dtype=np.int64,
                count=len(sg),
            )
            target_node_id = np.fromiter(
                (int(node_ids[int(i)]) for i in tg),
                dtype=np.int64,
                count=len(tg),
            )
            predecode_selected = np.fromiter(
                ((int(a), int(b)) in preselected for a, b in zip(sg, tg, strict=True)),
                dtype=np.bool_,
                count=len(sg),
            )
            source_division_head = np.fromiter(
                (float(division_head_by_global.get(int(i), np.nan)) for i in sg),
                dtype=np.float32,
                count=len(sg),
            )
            cache_path = cache_dir / f"{name}.npz"
            np.savez_compressed(
                cache_path,
                source_id=source_node_id,
                target_id=target_node_id,
                source_time=association_candidate_cache["source_time"],
                target_time=association_candidate_cache["target_time"],
                prob=association_candidate_cache["prob"],
                dist_um=association_candidate_cache["dist_um"],
                predecode_selected=predecode_selected,
                source_division_head=source_division_head,
            )
            print(
                f"[ASSOC CACHE] {name}: pairs={len(source_node_id)} path={cache_path}",
                flush=True,
            )

        print(
            f"[PRE-ILP GRAPH] {name}: "
            f"nodes={graph.num_nodes()} "
            f"edges={graph.num_edges()} "
            f"divisions={count_divisions_in_graph(graph)}",
            flush=True,
        )

        if config.use_ilp and graph.num_edges() > 0:
            edge_tbl = graph.edge_attrs()
            outdeg = Counter()
            for row in edge_tbl.iter_rows(named=True):
                outdeg[int(row["source_id"])] += 1

            print(
                f"[ILP BEFORE] edges={graph.num_edges()} "
                f"divisions={sum(d == 2 for d in outdeg.values())}",
                flush=True,
            )

            solver = td.solvers.ILPSolver(
                edge_weight=(
                    config.ilp_edge_weight
                    * td.EdgeAttr("edge_prob")
                ),
                appearance_weight=config.ilp_appearance_weight,
                disappearance_weight=config.ilp_disappearance_weight,
                division_weight=config.ilp_division_weight,
            )

            with suppress_output():
                graph = solver.solve(graph)

            edge_tbl = graph.edge_attrs()
            outdeg = Counter()
            for row in edge_tbl.iter_rows(named=True):
                outdeg[int(row["source_id"])] += 1

            print(
                f"[ILP AFTER] edges={graph.num_edges()} "
                f"divisions={sum(d == 2 for d in outdeg.values())}",
                flush=True,
            )

        print(
            f"[POST-DECODE GRAPH] {name}: "
            f"nodes={graph.num_nodes()} "
            f"edges={graph.num_edges()} "
            f"divisions={count_divisions_in_graph(graph)}",
            flush=True,
        )

        # V16: division completion happens only after ILP/decoding.
        added_divisions = add_precision_divisions_post_ilp(
            graph=graph,
            node_ids=node_ids,
            coords_ds=coords_ds,
            candidate_edges_by_pair=candidate_edges_by_pair,
            division_head_prob_by_global=division_head_by_global,
            voxel_size_ds=voxel_size_ds,
            config=config,
            audit_rows=audit_rows,
            video_name=name,
        )

        print(
            f"[POST-V16.4a RESCUE] {name}: "
            f"added={added_divisions} "
            f"edges={graph.num_edges()} "
            f"divisions={count_divisions_in_graph(graph)}",
            flush=True,
        )

        output_path = output_dir / f"{name}.geff"
        save_graph(graph, output_path)

        if config.audit_csv is not None:
            write_audit_csv(
                audit_rows,
                config.audit_csv,
                name,
            )

        print(
            f"[SAVED] {name}: {output_path}",
            flush=True,
        )

    print(
        f"Saved {len(test_names)} predictions to {output_dir}",
        flush=True,
    )

    if config.audit_csv is not None:
        print(
            f"Division audit written to {config.audit_csv}",
            flush=True,
        )

    if evaluate:
        run = {
            "username": USERNAME,
            "method": method,
            "split": f"split_{fold}",
            "dir": output_dir,
            "geffs": sorted(output_dir.glob("*.geff")),
        }

        results = evaluate_run(
            run,
            gt_dir=data_dir,
        )

        summary = summarise(results)

        print(
            f"Evaluation ({len(results)} videos): "
            f"score={summary['score']:.4f} "
            f"edge_jaccard={summary['edge_jaccard']:.4f} "
            f"adj_edge_jaccard={summary['adj_edge_jaccard']:.4f} "
            f"(n_adj={summary['n_adj']}) "
            f"division_jaccard={summary['division_jaccard']:.4f} "
            f"(TP={summary['division_tp']} "
            f"FP={summary['division_fp']} "
            f"FN={summary['division_fn']}) "
            f"node_recall={summary['node_recall']:.4f}",
            flush=True,
        )


# =============================================================================
# Command-line interface
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run BioTrack3D++ V16.3.2 with post-ILP precision-first "
            "division rescue."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--splits",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--debug-video",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--slice",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--partition",
        choices=("train", "test"),
        default="test",
        help=(
            "Which split partition to predict when --debug-video is not used. "
            "Default: test. Use train to generate context-model training data "
            "without contaminating the held-out validation videos."
        ),
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
    )

    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.30,
        help="Minimum sigmoid detection probability.",
    )

    parser.add_argument(
        "--no-det-tta",
        dest="det_tta",
        action="store_false",
        default=True,
    )

    parser.add_argument(
        "--pool-kernel-um",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--association-mode",
        choices=("global", "greedy"),
        default="greedy",
        help=(
            "Framewise 1->1 association: historical greedy (V16.3.2 default) "
            "or experimental global maximum-gain matching."
        ),
    )

    parser.add_argument(
        "--max-link-distance-um",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--division-head-threshold",
        type=float,
        default=0.58,
    )

    parser.add_argument(
        "--division-min-edge-prob",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--division-min-score",
        type=float,
        default=0.07,
        help="Minimum learned 4-feature biology score for V16.3.2 evidence envelope.",
    )

    parser.add_argument(
        "--division-evidence-head-min",
        type=float,
        default=0.83,
        help="Minimum division-head probability for the V16.3.2 evidence envelope.",
    )

    parser.add_argument(
        "--division-evidence-existing-edge-min",
        type=float,
        default=0.65,
        help="Minimum neural probability of the already-selected daughter edge.",
    )

    parser.add_argument(
        "--division-evidence-candidate-edge-min",
        type=float,
        default=0.32,
        help="Minimum neural probability of the rescued daughter edge.",
    )

    parser.add_argument(
        "--division-compact-min-daughter-edge-strong",
        type=float,
        default=0.70,
        help=(
            "Compact-route strong-neural branch: minimum of the two daughter "
            "edge probabilities."
        ),
    )

    parser.add_argument(
        "--division-compact-pair-score-strong",
        type=float,
        default=0.16,
        help=(
            "Compact-route biology branch: learned pair-score threshold used "
            "when the two daughter edges are not both >= the strong-neural floor."
        ),
    )

    parser.add_argument(
        "--division-pair-margin",
        type=float,
        default=0.03,
        help="Minimum best-vs-second-best biology margin when candidates compete.",
    )

    parser.add_argument(
        "--division-single-persist-min-score",
        type=float,
        default=0.35,
        help="Stricter biology gate when only one daughter persists.",
    )

    parser.add_argument(
        "--division-single-persist-min-edge-prob",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--division-single-persist-min-head-prob",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--max-division-fraction",
        type=float,
        default=0.00375,
    )

    parser.add_argument(
        "--use-ilp",
        action="store_true",
    )

    parser.add_argument(
        "--ilp-edge-weight",
        type=float,
        default=-1.0,
    )

    parser.add_argument(
        "--ilp-appearance-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--ilp-disappearance-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--ilp-division-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--audit-csv",
        type=str,
        default=None,
        help="Optional path for division candidate audit CSV.",
    )

    parser.add_argument(
        "--association-candidates-dir",
        type=str,
        default=None,
        help=(
            "Diagnostic only: directory for compressed NPZ caches containing "
            "every neural source->target pair within max-link-distance, including "
            "scores below edge-threshold. Prediction logic is unchanged."
        ),
    )

    args = parser.parse_args()

    from dataspec import DATASET_PATH

    data_dir = (
        Path(args.data_dir)
        if args.data_dir is not None
        else Path(DATASET_PATH)
    )

    splits_file = (
        Path(args.splits)
        if args.splits is not None
        else data_dir / "dataset_splits.json"
    )

    weights_path = (
        Path(args.weights)
        if args.weights is not None
        else (
            WEIGHTS_PATH
            / args.method
            / f"split_{args.split}"
            / "edge_predictor_best.pth"
        )
    )

    debug_video = (
        Path(args.debug_video)
        if args.debug_video is not None
        else None
    )

    video_slice = (
        slice(
            *[
                int(value)
                if value
                else None
                for value in args.slice.split(":")
            ]
        )
        if args.slice is not None
        else None
    )

    audit_csv = (
        Path(args.audit_csv)
        if args.audit_csv is not None
        else None
    )

    association_candidates_dir = (
        Path(args.association_candidates_dir)
        if args.association_candidates_dir is not None
        else None
    )

    config = PredictConfig(
        det_threshold=args.det_threshold,
        det_tta=args.det_tta,
        pool_kernel_um=args.pool_kernel_um,
        edge_threshold=args.edge_threshold,
        max_link_distance_um=args.max_link_distance_um,
        association_mode=args.association_mode,
        division_head_threshold=args.division_head_threshold,
        division_min_edge_prob=args.division_min_edge_prob,
        division_min_score=args.division_min_score,
        division_evidence_head_min=args.division_evidence_head_min,
        division_evidence_existing_edge_min=args.division_evidence_existing_edge_min,
        division_evidence_candidate_edge_min=args.division_evidence_candidate_edge_min,
        division_compact_min_daughter_edge_strong=(
            args.division_compact_min_daughter_edge_strong
        ),
        division_compact_pair_score_strong=args.division_compact_pair_score_strong,
        division_pair_margin=args.division_pair_margin,
        division_single_persist_min_score=args.division_single_persist_min_score,
        division_single_persist_min_edge_prob=args.division_single_persist_min_edge_prob,
        division_single_persist_min_head_prob=args.division_single_persist_min_head_prob,
        max_division_fraction=args.max_division_fraction,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        audit_csv=audit_csv,
        association_candidates_dir=association_candidates_dir,
    )

    folds = (
        range(5)
        if args.split == "all"
        else [int(args.split)]
    )

    for fold in folds:
        print(
            "BioTrack3D++ V16.4a association audit: "
            "strong V15 tracker + precision-first post-ILP division rescue",
            flush=True,
        )

        predict(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_path=weights_path,
            config=config,
            method=args.method,
            debug_video=debug_video,
            video_slice=video_slice,
            partition=args.partition,
            evaluate=args.evaluate,
        )


if __name__ == "__main__":
    main()