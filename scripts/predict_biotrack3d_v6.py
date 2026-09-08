#!/usr/bin/env python
"""BioTrack3D++ V15: Post-ILP Learned Biological Rescue

Features:
- Preserves the V13 detector, neural edge scores, greedy candidate construction,
  and ILP settings that produced the strong V13 tracking baseline.
- Keeps V13's conservative pre-ILP family-completion proposals unchanged.
- Adds a final global post-ILP missing-daughter rescue, so ILP cannot delete a
  biologically accepted second-daughter edge after it is added.
- Uses the BioHub-derived learned mitosis score on its calibrated logistic scale
  with a post-ILP acceptance threshold of 0.20.
- Requires parent track continuity, an orphan second daughter, and one-step
  persistence of both daughter lineages before adding a division edge.

Usage:
    python scripts/predict_biotrack3d_v14.py --split 0
"""

from networkx.algorithms.connectivity import disjoint_paths
import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

import tracksdata as td

from tracking_cellmot.io import open_dataset, save_graph

# Import model and helpers from companion training script.
sys.path.insert(0, str(Path(__file__).parent))
from train_biotrack3d_v3 import (
    DEFAULT_METHOD,
    UNetNodeTransformer,
    extract_pos_features,
    _POS_EMBED_DIM,
)
from tracking_cellmot.models import TemporalUNet3D

from dataspec import USERNAME, INTERACTIVE, WEIGHTS_PATH
from evaluate import evaluate_run
from tracking_cellmot.metrics import summarise


# =============================================================================
# Prediction config
# =============================================================================

@dataclass
class PredictConfig:
    """All hyperparameters that can affect prediction quality / score.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"sigmoid"`` (independent
        per-edge scores) or ``"softmax"`` (row-normalised over t+1 nodes).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """
    # Detection
    det_threshold: float = 0.3
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 5.0  # max-pool kernel size in µm for detection peak extraction
    # Edge filtering
    edge_activation: str = "sigmoid"  # "sigmoid" or "softmax"
    threshold: float = 0.5

    # ILP post-processing
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # When ILP is enabled it handles parent/children constraints itself,
        # so greedy limits are left unconstrained (None).  When ILP is
        # disabled, default to 1/1 to avoid unconstrained edge assignment.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2



# =============================================================================
# Helpers
# =============================================================================


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

# =============================================================================
# Graph building
# =============================================================================

def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> tuple[td.graph.InMemoryGraph, list[int]]:
    """Build a tracksdata graph and return its positional->internal node-ID map.

    Avoids ``add_node_attr_key`` to sidestep a tracksdata/Polars compatibility
    issue where the float default value is mistakenly used as a dtype.
    Probabilities are passed as-is; the normal inference path uses sigmoid
edge probabilities in [0, 1].
    """
    graph = td.graph.InMemoryGraph()

    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])
    return graph, list(node_ids)


# =============================================================================
# Model loading
# =============================================================================

_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct UNetNodeTransformer from saved config + weights.

    Reads ``config.json`` from the same directory as the weights file.
    Falls back to ``_DEFAULT_CONFIG`` if the file is missing.

    Returns ``(model, window_size, downsample)``.
    """
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        config = {**_DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.", flush=True)
        config = _DEFAULT_CONFIG

    # Support legacy configs that used "downsample_factor" (scalar).
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]

    downsample = tuple(config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
        pooling_mode=config.get("pooling_mode", "single_voxel"),
        pool_radius=config.get("pool_radius", 0),
        pool_sigma=config.get("pool_sigma", None),
        attn_dim=config.get("attn_dim", None),
        sibling_aware=config.get("sibling_aware", True),
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config["window_size"], downsample

# Division-aware post-processing.
# Kept separate from the neural edge predictor so we can test the
# biological hypothesis without retraining the model.

# =============================================================================
# Division-aware graph analysis
# =============================================================================

def find_division_candidates(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    downsample: tuple[int, ...] = (1, 1, 1),
    min_edge_prob: float = 0.5,
    parent_max_um: float = 12.0,
    daughter_min_um: float = 5.0,
    daughter_max_um: float = 13.0,
    min_persistence_prob: float = 0.3,
) -> list[dict]:
    """Find biologically plausible 1 -> 2 division events.

    A division candidate requires:

    1. One parent at frame t.
    2. Two distinct daughters at frame t+1.
    3. Both parent->daughter edges have reasonable probability.
    4. Both parent->daughter distances are biologically plausible.
    5. The daughters are spatially separated but still plausibly siblings.
    6. Both daughters have evidence of continuing into t+2.

    This is inference-time analysis only. It does not modify the neural model.
    """

    if not edges or len(coords) == 0:
        return []
    ds = np.asarray(downsample, dtype=np.float32)

    # ------------------------------------------------------------------
    # Organise edges by source node.
    # ------------------------------------------------------------------
    outgoing: dict[int, list[tuple[int, float, float]]] = {}

    for src, tgt, prob, dist in edges:
        if prob < min_edge_prob:
            continue
        outgoing.setdefault(src, []).append(
            (tgt, float(prob), float(dist))
        )

    # Fast lookup for whether a node has a plausible outgoing continuation.
    continuation_prob: dict[int, float] = {}

    for src, tgt, prob, dist in edges:
        if prob < min_persistence_prob:
            continue

        previous = continuation_prob.get(src, 0.0)
        continuation_prob[src] = max(previous, float(prob))

    candidates: list[dict] = []

    # ------------------------------------------------------------------
    # Examine each parent with at least two candidate children.
    # ------------------------------------------------------------------
    for parent_idx, child_edges in outgoing.items():
        if len(child_edges) < 2:
            continue

        parent = coords[parent_idx]
        parent_t = int(parent[0])

        # Only consider candidates in the immediately following frame.
        valid_children = []

        for child_idx, prob, dist_from_edge in child_edges:
            child = coords[child_idx]

            if int(child[0]) != parent_t + 1:
                continue

            # Recompute physical parent->child distance from coordinates.
            # coords are [t, z, y, x] in original-resolution voxel units.
            delta = (
                np.asarray(parent[1:], dtype=np.float32)
                - np.asarray(child[1:], dtype=np.float32)
            )

            physical_delta = delta * ds * np.asarray(
                [1.625, 0.40625, 0.40625],
                dtype=np.float32,
            )

            physical_dist = float(np.linalg.norm(physical_delta))

            if physical_dist > parent_max_um:
                continue

            valid_children.append(
                {
                    "idx": child_idx,
                    "prob": float(prob),
                    "dist": physical_dist,
                }
            )

        if len(valid_children) < 2:
            continue

        # --------------------------------------------------------------
        # Examine every possible daughter pair.
        # --------------------------------------------------------------
        for a in range(len(valid_children)):
            for b in range(a + 1, len(valid_children)):
                da = valid_children[a]
                db = valid_children[b]

                child_a = coords[da["idx"]]
                child_b = coords[db["idx"]]

                daughter_delta = (
                    np.asarray(child_a[1:], dtype=np.float32)
                    - np.asarray(child_b[1:], dtype=np.float32)
                )

                physical_daughter_delta = daughter_delta * ds * np.asarray(
                    [1.625, 0.40625, 0.40625],
                    dtype=np.float32,
                )

                daughter_distance = float(
                    np.linalg.norm(physical_daughter_delta)
                )

                if not (
                    daughter_min_um
                    <= daughter_distance
                    <= daughter_max_um
                ):
                    continue

                # ------------------------------------------------------
                # Symmetry: daughter distances from parent should be
                # reasonably similar.
                # ------------------------------------------------------
                d1 = da["dist"]
                d2 = db["dist"]

                symmetry = abs(d1 - d2) / max(d1 + d2, 1e-6)

                if symmetry > 0.75:
                    continue

                # ------------------------------------------------------
                # Midpoint geometry.
                # ------------------------------------------------------
                midpoint = (
                    np.asarray(child_a[1:], dtype=np.float32)
                    + np.asarray(child_b[1:], dtype=np.float32)
                ) / 2.0

                midpoint_delta = (
                    np.asarray(parent[1:], dtype=np.float32)
                    - midpoint
                )

                midpoint_physical = midpoint_delta * ds * np.asarray(
                    [1.625, 0.40625, 0.40625],
                    dtype=np.float32,
                )

                midpoint_distance = float(
                    np.linalg.norm(midpoint_physical)
                )

                # ------------------------------------------------------
                # Daughter persistence into the following frame.
                #
                # We look for ANY plausible outgoing edge from each
                # daughter. This is evidence that the daughter survives
                # beyond the immediate division frame.
                # ------------------------------------------------------
                persist_a = continuation_prob.get(da["idx"], 0.0)
                persist_b = continuation_prob.get(db["idx"], 0.0)

                # ------------------------------------------------------
                # Combined interpretable division score.
                # ------------------------------------------------------
                edge_score = 0.5 * (da["prob"] + db["prob"])

                persistence_score = 0.5 * (
                    persist_a + persist_b
                )

                geometry_score = (
                    1.0
                    - min(symmetry, 1.0)
                )

                division_score = (
                    0.5 * edge_score
                    + 0.3 * persistence_score
                    + 0.2 * geometry_score
                )

                candidates.append(
                    {
                        "parent": int(parent_idx),
                        "daughter1": int(da["idx"]),
                        "daughter2": int(db["idx"]),

                        "parent_coord": np.r_[
                            coords[parent_idx, 0],
                            coords[parent_idx, 1:] * ds,
                        ].tolist(),

                        "daughter1_coord": np.r_[
                            coords[da["idx"], 0],
                            coords[da["idx"], 1:] * ds,
                        ].tolist(),

                        "daughter2_coord": np.r_[
                            coords[db["idx"], 0],
                            coords[db["idx"], 1:] * ds,
                        ].tolist(),

                        "edge_prob1": float(da["prob"]),
                        "edge_prob2": float(db["prob"]),
                        "parent_dist1_um": float(d1),
                        "parent_dist2_um": float(d2),
                        "daughter_dist_um": float(daughter_distance),
                        "symmetry": float(symmetry),
                        "midpoint_dist_um": float(midpoint_distance),
                        "persistence1": float(persist_a),
                        "persistence2": float(persist_b),
                        "division_score": float(division_score),
                    }
                )
    # Highest-confidence division hypotheses first.
    candidates.sort(
        key=lambda x: x["division_score"],
        reverse=True,
    )

    return candidates

def score_mitosis_pair(
    edge1,
    edge2,
    dist1,
    dist2,
    sister_dist,
    midpoint_error,
    angle_deg,
):
    """
    Atlas-derived biological mitosis score.

    Returns
    -------
    float
        Final pair score, or -1 if the pair violates
        biological constraints.
    """

    # --------------------------------------------------
    # Hard biological gates (learned from GT atlas)
    # --------------------------------------------------

    if dist1 > 11.0 or dist2 > 11.0:
        return -1.0

    if not (7.0 <= sister_dist <= 15.3):
        return -1.0

    if midpoint_error > 6.0:
        return -1.0

    # Parent Distance Symmetry (PDS)
    pds = min(dist1, dist2) / max(dist1, dist2)


    # --------------------------------------------------
    # Learned BioHub biological score
    # (151 real mitoses vs 5403 false candidates)
    # --------------------------------------------------

    # Convert our runtime angle score back to degrees.
    # Standardize using the learned atlas statistics.
    z_angle = (
        (angle_deg - 103.11636742361918)
        / 39.635689953206395
    )
    z_pds = (pds - 0.2606685995216205) / 0.21502299096448316
    z_sister = (sister_dist - 9.773215468212301) / 2.2774296624182515
    z_mid = (midpoint_error - 2.3459422462624873) / 0.27467692920998715

    # Linear score (intercept omitted; ranking is unchanged).
    z = (
        -5.499524582405182
        + 1.7994280408871015 * z_angle
        + 1.332141158603924 * z_pds
        - 0.5896250937667038 * z_sister
        + 0.5635888501431774 * z_mid
    )

    # Convert to a smooth ranking score.
    score = 1.0 / (1.0 + np.exp(-z))

    return float(score)
# =============================================================================
# Per-frame loading
# =============================================================================

def _load_frame(
    zarr_arr,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    """Load one frame from zarr with strided spatial downsample (no normalisation)."""
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape,
            mode="trilinear", align_corners=False,
        )[0, 0]
    return frame


# =============================================================================
# Inference
# =============================================================================

def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd
    (for symmetric padding) and at least 1.

    Parameters
    ----------
    um : float
        Desired suppression distance in microns.
    voxel_size : tuple[float, ...]
        Per-axis voxel sizes in microns, e.g. ``(1.625, 0.40625, 0.40625)``.
    """
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.3,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Coordinates are returned in the downsampled grid.  The caller is
    responsible for scaling back to original resolution if needed.

    Parameters
    ----------
    det_logits : torch.Tensor
        (1, Z, Y, X) raw logits.
    t : int
        Time index to prepend as the first column.
    det_threshold : float
        Minimum logit for a peak to be considered (default 0.3).
    pool_kernel : tuple[int, ...]
        Per-axis kernel size for local-max pooling,
        e.g. ``(3, 11, 11)`` for anisotropic data.

    Returns
    -------
    np.ndarray
        (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)

    # Detection threshold is defined as a probability, so convert logits
    # before thresholding.
    probs = torch.sigmoid(logits)

    pad = tuple(
        k // 2
        for k in pool_kernel
    )

    pooled = F.max_pool3d(
        probs,
        pool_kernel,
        stride=1,
        padding=pad,
    )

    is_peak = (
        (probs == pooled)
        & (probs >= det_threshold)
    )

    peak_idx = torch.nonzero(
        is_peak[0, 0]
    )
    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
    use_gt_coords: bool = False,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, float, float]],
    dict[int, float],
    dict[tuple[int, int], float],
]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once. UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
    division_head_by_global : dict[int, float]
        Maximum predicted division-head probability observed for each global node.
    raw_edge_prob_by_pair : dict[tuple[int, int], float]
        Raw neural edge probability for local parent->candidate pairs.
    """
    ds = open_dataset(
        ds_path,
        normalize=False,
        load_image=False,
        downsample=downsample,
        require_tracks=use_gt_coords,
    )

    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(
            f"Zarr attrs missing image_statistics.quantiles for {ds_path}"
        )

    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]

    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = (
        ds.image_shape[0]
        if max_frames is None
        else min(ds.image_shape[0], max_frames)
    )

    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])

    ds_arr = np.array(
        downsample,
        dtype=np.float32,
    )
    # ----------------------------------------------------------
    # Candidate atlas logger.
    # ----------------------------------------------------------
    candidate_csv = open(
        "candidate_atlas.csv",
        "w",
        newline="",
    )

    candidate_writer = csv.writer(candidate_csv)

    candidate_writer.writerow([
    "video",
    "frame",
    "parent_local",
    "child1_local",
    "child2_local",
    "parent_global",
    "child1_global",
    "child2_global",
    "edge1",
    "edge2",
    "parent_d1",
    "parent_d2",
    "sister_distance",
    "midpoint_error",
    "pds",
    "angle_deg",
    "pair_score",
    "selected",
    ])

    ds_arr_t = torch.from_numpy(ds_arr).to(device)

    pos_feat_dim = 4 * _POS_EMBED_DIM

    W = window_size

    voxel_size = tuple(
        s * d
        for s, d in zip(ds.scale, downsample)
    )

    pool_k = pool_kernel_from_um(
        cfg.pool_kernel_um,
        voxel_size,
    )

    # ------------------------------------------------------------------
    # Running node registry.
    # ------------------------------------------------------------------
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()

    coord_lists: list[np.ndarray] = []

    coord_offset: dict[int, tuple[int, int]] = {}

    global_node_count: int = 0

    all_edges: list[
        tuple[int, int, float, float]
    ] = []

    # Parent-level division-head probabilities keyed by global node index.
    # Stored so the final post-ILP rescue can use the neural division prior.
    division_head_by_global: dict[int, float] = {}

    # Raw neural parent->candidate probabilities for the final post-ILP rescue.
    # V14 scanned every spatially plausible orphan, which created geometry-only
    # false rescue proposals. V15 requires direct neural compatibility too.
    raw_edge_prob_by_pair: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Window layout.
    # ------------------------------------------------------------------
    stride = max(W - 1, 1)

    window_starts = list(
        range(
            0,
            T - W + 1,
            stride,
        )
    )

    # Ensure the very last pair (T-2 -> T-1) is covered.
    if (
        not window_starts
        or window_starts[-1] + W < T
    ):
        last = max(T - W, 0)

        if (
            not window_starts
            or last != window_starts[-1]
        ):
            window_starts.append(last)

    # ==================================================================
    # Sliding-window inference.
    # ==================================================================
    for ws in tqdm(
        window_starts,
        desc="  windows",
        leave=False,
        disable=not INTERACTIVE,
    ):
        frame_indices = list(
            range(ws, ws + W)
        )

        # --------------------------------------------------------------
        # UNet encode.
        # --------------------------------------------------------------
        imgs = torch.stack(
            [
                _load_frame(
                    zarr_arr,
                    t,
                    target_shape,
                    downsample,
                )
                for t in frame_indices
            ]
        )

        # Quantile normalization.
        imgs = (
            (imgs - q_low)
            / (q_high - q_low + 1e-6)
        ).clamp(0.0)

        imgs = imgs.unsqueeze(0).to(device)

        unet_out, det_logits = model.encode(imgs)

        # --------------------------------------------------------------
        # Detection TTA.
        # --------------------------------------------------------------
        if cfg.det_tta:
            tta_flips = [
                (-1,),
                (-2,),
                (-2, -1),
            ]

            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)

                _, det_flip = model.encode(
                    imgs_flip
                )

                for f in range(W):
                    det_logits[f] = (
                        det_logits[f]
                        + det_flip[f].flip(dims)
                    )

                del imgs_flip
                del det_flip

            for f in range(W):
                det_logits[f] = (
                    det_logits[f] / 4
                )

        del imgs

        # --------------------------------------------------------------
        # Detect cells in each frame.
        # --------------------------------------------------------------
        for f_idx, t in enumerate(frame_indices):

            if t not in seen_frames:

                if use_gt_coords:
                    import polars as pl

                    gt_t = (
                        ds.tracks
                        .node_attrs()
                        .filter(
                            pl.col("t") == t
                        )
                    )

                    coords_t = (
                        gt_t
                        .select(["z", "y", "x"])
                        .to_numpy()
                        .astype(np.float32)
                        / ds_arr
                    )

                    t_col = np.full(
                        (len(coords_t), 1),
                        t,
                        dtype=np.float32,
                    )

                    arr = np.concatenate(
                        [
                            t_col,
                            coords_t,
                        ],
                        axis=1,
                    ).astype(np.float32)

                else:
                    arr = _detect_cells_pooled(
                        det_logits[f_idx][0],
                        t,
                        cfg.det_threshold,
                        pool_k,
                    )
                if len(arr) > 1:
                    det_frame = det_logits[f_idx][0, 0]

                    peak_scores = det_frame[
                        arr[:, 1],
                        arr[:, 2],
                        arr[:, 3],
                    ].detach().cpu().numpy()
                    keep = np.ones(len(arr), dtype=bool)

                    # Conservative same-frame deduplication.
                    # Coordinates are still in downsampled voxel space here.
                    dedup_radius_um = 3.5

                    spatial_scale_um = np.asarray(
                        [1.625, 0.40625, 0.40625],
                        dtype=np.float32,
                    ) * ds_arr

                    order = np.argsort(
                        peak_scores
                    )[::-1]

                    kept_indices = []

                    for idx in order:
                        if not keep[idx]:
                            continue

                        kept_indices.append(idx)

                        delta_voxel = (
                            arr[:, 1:].astype(np.float32)
                            - arr[idx, 1:].astype(np.float32)
                        )

                        delta_um = (
                            delta_voxel
                            * spatial_scale_um
                        )

                        dist_um = np.linalg.norm(
                            delta_um,
                            axis=1,
                        )

                        duplicate_mask = (
                            dist_um <= dedup_radius_um
                        )

                        duplicate_mask[idx] = False
                        keep[duplicate_mask] = False

                    arr = arr[
                        np.array(
                            sorted(kept_indices),
                            dtype=np.int64,
                        )
                    ]

                coord_offset[t] = (
                    global_node_count,
                    global_node_count + len(arr),
                )

                global_node_count += len(arr)

                coord_lists.append(arr)

                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists)
            if coord_lists
            else np.empty(
                (0, 4),
                dtype=np.int16,
            )
        )

        # ==============================================================
        # Edge prediction for consecutive frame pairs.
        # ==============================================================
        for f_idx in range(W - 1):

            t_src = frame_indices[f_idx]
            t_tgt = frame_indices[f_idx + 1]

            if (
                t_src,
                t_tgt,
            ) in seen_pairs:
                continue

            seen_pairs.add(
                (
                    t_src,
                    t_tgt,
                )
            )

            if (
                t_src not in coord_offset
                or t_tgt not in coord_offset
            ):
                continue

            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]

            if (
                e_src == s_src
                or e_tgt == s_tgt
            ):
                continue

            c_src = coords_so_far[
                s_src:e_src
            ]

            c_tgt = coords_so_far[
                s_tgt:e_tgt
            ]

            n_src = len(c_src)
            n_tgt = len(c_tgt)

            idx_src = np.arange(
                s_src,
                e_src,
                dtype=np.int64,
            )

            idx_tgt = np.arange(
                s_tgt,
                e_tgt,
                dtype=np.int64,
            )

            # ----------------------------------------------------------
            # Build tensors.
            # ----------------------------------------------------------
            p_coords_src = (
                torch.from_numpy(
                    c_src[:, 1:]
                    .astype(np.float32)
                )
                .unsqueeze(0)
                .to(device)
            )

            p_coords_tgt = (
                torch.from_numpy(
                    c_tgt[:, 1:]
                    .astype(np.float32)
                )
                .unsqueeze(0)
                .to(device)
            )

            # Use window-relative time.
            window_shape = (
                W,
            ) + image_shape[1:]

            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx

            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1

            p_pos_src = (
                torch.from_numpy(
                    extract_pos_features(
                        c_src_rel,
                        window_shape,
                    )
                )
                .unsqueeze(0)
                .to(device)
            )

            p_pos_tgt = (
                torch.from_numpy(
                    extract_pos_features(
                        c_tgt_rel,
                        window_shape,
                    )
                )
                .unsqueeze(0)
                .to(device)
            )

            p_mask_src = torch.ones(
                1,
                n_src,
                dtype=torch.bool,
                device=device,
            )

            p_mask_tgt = torch.ones(
                1,
                n_tgt,
                dtype=torch.bool,
                device=device,
            )

            # ----------------------------------------------------------
            # Index UNet features.
            # ----------------------------------------------------------
            unet_feat_src = model._index_features(
                unet_out[:, f_idx],
                p_coords_src,
                p_mask_src,
            )

            unet_feat_tgt = model._index_features(
                unet_out[:, f_idx + 1],
                p_coords_tgt,
                p_mask_tgt,
            )

            # ----------------------------------------------------------
            # Predict edge logits.
            # ----------------------------------------------------------
            edge_logits_pair = model.predict_edges(
                unet_feat_src,
                unet_feat_tgt,
                p_coords_src * ds_arr_t,
                p_coords_tgt * ds_arr_t,
                p_pos_src,
                p_pos_tgt,
                p_mask_src,
                p_mask_tgt,
            )

            # ----------------------------------------------------------
            # Parent-level division prediction.
            # This uses the trained division head with the exact same
            # feature/coordinate conventions used during training.
            # ----------------------------------------------------------
            division_logits_pair = model.predict_divisions(
                unet_feat_src,
                unet_feat_tgt,
                p_coords_src * ds_arr_t,
                p_coords_tgt * ds_arr_t,
                p_pos_src,
                p_pos_tgt,
                p_mask_src,
                p_mask_tgt,
            )

            division_probs = (
                torch.sigmoid(
                    division_logits_pair[0]
                )
                .detach()
                .cpu()
                .numpy()
            )

            # Keep the strongest division-head probability observed for each
            # source node across windows/frame-pair evaluations.
            for local_i, div_prob in enumerate(division_probs):
                global_i = int(idx_src[local_i])
                division_head_by_global[global_i] = max(
                    division_head_by_global.get(global_i, 0.0),
                    float(div_prob),
                )

            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = (
                    torch.softmax(
                        raw,
                        dim=0,
                    )
                    .cpu()
                    .numpy()
                )
            else:
                probs = (
                    torch.sigmoid(raw)
                    .cpu()
                    .numpy()
                )

            # ==========================================================
            # M2: conservative mitosis-aware association.
            # ==========================================================
            MITOSIS_PARENT_MAX_UM = 12.0
            MITOSIS_DAUGHTER_MIN_UM = 5.0
            MITOSIS_DAUGHTER_MAX_UM = 13.0
            MITOSIS_BONUS = 0.05

            # Require a clear improvement before replacing a daughter.
            MITOSIS_REPLACEMENT_MARGIN = 0.05

            MITOSIS_MIN_EDGE_PROB = cfg.threshold

            # p_coords_* are (1, n_nodes, 3).
            # These are original-resolution voxel coordinates.
            src_xyz = (
                (p_coords_src * ds_arr_t)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            tgt_xyz = (
                (p_coords_tgt * ds_arr_t)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )


            # ==========================================================
            # Build normal neural candidates.
            # ==========================================================
            candidate_data = []

            for i in range(n_src):
                for j in range(n_tgt):

                    prob = float(
                        probs[i, j]
                    )

                    # ----------------------------------------------------------
                    # Compute physical distance BEFORE thresholding.
                    # This allows a mitosis-specific rescue.
                    # ----------------------------------------------------------
                    delta_voxel = (
                        src_xyz[i]
                        - tgt_xyz[j]
                    )

                    physical_delta = (
                        delta_voxel
                        * np.asarray(
                            [1.625, 0.40625, 0.40625],
                            dtype=np.float32,
                        )
                    )

                    dist = float(
                        np.linalg.norm(physical_delta)
                    )

                    # V15: preserve direct neural parent->daughter evidence for
                    # physically local pairs so final rescue is not geometry-only.
                    if dist <= 12.0:
                        raw_key = (int(idx_src[i]), int(idx_tgt[j]))
                        raw_edge_prob_by_pair[raw_key] = max(
                            raw_edge_prob_by_pair.get(raw_key, 0.0),
                            prob,
                        )

                    # ----------------------------------------------------------
                    # Mitosis-specific candidate expansion.
                    # ----------------------------------------------------------
                    is_dividing_parent = (
                        float(division_probs[i]) > 0.30
                    )

                    if is_dividing_parent:
                        # Allow weaker secondary daughter edges,
                        # but only if they stay physically close.
                        if prob <= 0.15 and dist > 12.0:
                            continue
                    else:
                        # Keep normal tracking unchanged.
                        if prob <= cfg.threshold:
                            continue

                    # Keep the original logic unchanged.
                    candidate_data.append(
                        {
                            "prob": prob,
                            "i": i,
                            "j": j,
                            "dist": dist,
                            "score": prob,
                        }
                    )
            
            # ==========================================================
            # 1. Standard Greedy 1->1 Association
            # ==========================================================
            candidates = sorted(
                [
                    (
                        c["score"],
                        c["prob"],
                        c["i"],
                        c["j"],
                        c["dist"],
                    )
                    for c in candidate_data
                ],
                reverse=True,
            )

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}

            for (
                score,
                prob,
                i,
                j,
                dist,
            ) in candidates:

                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)

                # Standard 1->1 tracking: 1 child per parent during initial greedy pass.
                if n_ch >= 1:
                    continue

                if (
                    cfg.max_parents_per_node is not None
                    and n_pa >= cfg.max_parents_per_node
                ):
                    continue

                gi = int(idx_src[i])
                gj = int(idx_tgt[j])

                all_edges.append(
                    (
                        gi,
                        gj,
                        float(prob),
                        float(dist),
                    )
                )

                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1

            # ==========================================================
            # 2. V13-preserving pre-ILP family-completion proposals.
            #    The 0.60 gate is intentionally retained here to preserve the
            #    V13 candidate graph seen by ILP. V15's calibrated 0.20 rescue
            #    is applied globally only after ILP has finished.
            # ==========================================================
            VOXEL = np.asarray(
                [1.625, 0.40625, 0.40625],
                dtype=np.float32,
            )
            rescue_proposals: list[dict] = []
            for parent_idx, n_children in list(children_count.items()):

                if n_children != 1:
                    continue

                gi = int(idx_src[parent_idx])

                # Find existing first daughter from all_edges
                existing_daughter_idx = None
                existing_prob = 0.0
                existing_dist = 0.0

                for src, tgt, prob, dist in all_edges:
                    if src == gi:
                        matches = np.where(idx_tgt == tgt)[0]
                        if len(matches) > 0:
                            existing_daughter_idx = int(matches[0])
                            existing_prob = float(prob)
                            existing_dist = float(dist)
                        break

                if existing_daughter_idx is None:
                    continue
                p_xyz = src_xyz[parent_idx]
                e_xyz = tgt_xyz[existing_daughter_idx]

                # Search candidate pool for the 2nd missing daughter
                for cand_j in range(n_tgt):
                    if cand_j == existing_daughter_idx:
                        continue

                    # Never select a daughter that already has a parent
                    if parents_count.get(cand_j, 0) != 0:
                        continue

                    cand_prob = float(probs[parent_idx, cand_j])
                    cand_xyz = tgt_xyz[cand_j]

                    delta_p = (p_xyz - cand_xyz) * VOXEL
                    cand_dist = float(np.linalg.norm(delta_p))

                    delta_s = (e_xyz - cand_xyz) * VOXEL
                    sister_dist = float(np.linalg.norm(delta_s))

                    midpoint = (e_xyz + cand_xyz) / 2.0
                    midpoint_delta = (midpoint - p_xyz) * VOXEL
                    midpoint_error = float(
                        np.linalg.norm(midpoint_delta)
                    )

                    v1 = (e_xyz - p_xyz) * VOXEL
                    v2 = (cand_xyz - p_xyz) * VOXEL
                    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                    if denom == 0:
                        continue

                    angle_deg = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(v1, v2) / denom,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )

                    pair_score = score_mitosis_pair(
                        edge1=existing_prob,
                        edge2=cand_prob,
                        dist1=existing_dist,
                        dist2=cand_dist,
                        sister_dist=sister_dist,
                        midpoint_error=midpoint_error,
                        angle_deg=angle_deg,
                    )

                    # Conservative acceptance gate
                    if pair_score < 0.60:
                        continue
                    rescue_proposals.append(
                        {
                            "score": float(pair_score),
                            "parent_idx": parent_idx,
                            "existing_daughter_idx": existing_daughter_idx,
                            "cand_j": cand_j,
                            "existing_prob": existing_prob,
                            "existing_dist": existing_dist,
                            "cand_prob": cand_prob,
                            "cand_dist": cand_dist,
                            "sister_dist": sister_dist,
                            "midpoint_error": midpoint_error,
                            "pds": min(existing_dist, cand_dist) / max(existing_dist, cand_dist, 1e-6),
                            "angle_deg": angle_deg,
                        }
                    )

            # Sort proposals by pair_score descending
            rescue_proposals.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            rescued_parents: set[int] = set()

            for prop in rescue_proposals:
                p_idx = prop["parent_idx"]
                c_j = prop["cand_j"]

                if p_idx in rescued_parents:
                    continue
                if children_count.get(p_idx, 0) != 1:
                    continue
                if parents_count.get(c_j, 0) != 0:
                    continue

                gi = int(idx_src[p_idx])
                gj2 = int(idx_tgt[c_j])

                all_edges.append(
                    (
                        gi,
                        gj2,
                        float(prop["cand_prob"]),
                        float(prop["cand_dist"]),
                    )
                )

                children_count[p_idx] = 2
                parents_count[c_j] = 1
                rescued_parents.add(p_idx)

                candidate_writer.writerow([
                    ds_path.stem,
                    t_src,
                    p_idx,
                    prop["existing_daughter_idx"],
                    c_j,
                    int(idx_src[p_idx]),
                    int(idx_tgt[prop["existing_daughter_idx"]]),
                    gj2,
                    prop["existing_prob"],
                    prop["cand_prob"],
                    prop["existing_dist"],
                    prop["cand_dist"],
                    prop["sister_dist"],
                    prop["midpoint_error"],
                    prop["pds"],
                    prop["angle_deg"],
                    prop["score"],
                    1,
                ])

        del unet_out
    candidate_csv.close()

    # ==============================================================
    # Diagnostic division analysis.
    # Does NOT modify all_edges.
    # ==============================================================
    division_candidates = find_division_candidates(
        coords=(
            np.concatenate(coord_lists)
            if coord_lists
            else np.empty(
                (0, 4)
            )
        ),
        edges=all_edges,
        downsample=tuple(
            int(x)
            for x in ds_arr
        ),
        min_edge_prob=0.5,
        parent_max_um=12.0,
        daughter_min_um=5.0,
        daughter_max_um=13.0,
        min_persistence_prob=0.3,
    )
    # ==============================================================
    # Final coordinate stack.
    # ==============================================================
    coords = (
        np.concatenate(coord_lists)
        if coord_lists
        else np.empty(
            (0, 4),
            dtype=np.int16,
        )
    )

    # Scale spatial coords back to original resolution.
    coords = coords.astype(
        np.float32
    )

    coords[:, 1:] *= ds_arr

    coords = coords.astype(
        np.int16
    )

    return coords, all_edges, division_head_by_global, raw_edge_prob_by_pair


# =============================================================================
# V15 final post-ILP biological rescue
# =============================================================================

def add_post_ilp_biological_rescue(
    graph: td.graph.InMemoryGraph,
    coords: np.ndarray,
    positional_node_ids: list[int],
    division_head_by_global: dict[int, float],
    raw_edge_prob_by_pair: dict[tuple[int, int], float],
    pair_score_threshold: float = 0.20,
    division_head_threshold: float = 0.30,
    min_candidate_edge_prob: float = 0.20,
    min_pair_margin: float = 0.0,
    max_division_fraction: float = 0.00375,
    debug_parent: int | None = None,
    debug_candidate: int | None = None,
) -> tuple[td.graph.InMemoryGraph, list[dict]]:
    """Add conservative missing-daughter edges *after* ILP has finished.

    This is the V15 correction to V14's sequencing.  V13 created biological
    rescue candidates before ILP, so the solver could still discard a true
    second-daughter edge.  V14 inspects the final ILP topology and only then
    adds one missing daughter to a parent when all of the following hold:

    - the parent already belongs to a track (has an incoming edge);
    - the parent has exactly one existing child at t+1;
    - the candidate second daughter is orphaned at t+1;
    - the raw neural parent->candidate edge probability is at least
      ``min_candidate_edge_prob``;
    - the neural division head is at least ``division_head_threshold``;
    - both daughter branches continue independently to t+2;
    - the BioHub learned geometry score is at least ``pair_score_threshold``;
    - if multiple candidates survive, the best score beats the runner-up by
      at least ``min_pair_margin``.

    The learned score is a calibrated logistic probability, so the V13 0.60
    heuristic gate is deliberately *not* reused here.
    """
    if len(coords) == 0 or graph.num_edges() == 0:
        return graph, []

    if len(positional_node_ids) != len(coords):
        raise ValueError(
            "positional_node_ids must have one entry for every coordinate"
        )

    # ILPSolver is expected to preserve node IDs.  Build an inverse map from
    # tracksdata's internal IDs back to the positional/global indices used by
    # the neural tracker and the learned mitosis atlas.
    internal_to_global = {
        int(node_id): global_idx
        for global_idx, node_id in enumerate(positional_node_ids)
    }
    current_node_ids = set(int(x) for x in graph.node_ids())
    missing_ids = [
        node_id for node_id in current_node_ids
        if node_id not in internal_to_global
    ]
    if missing_ids:
        raise RuntimeError(
            "ILP changed tracksdata node IDs; cannot safely apply the V14 "
            "post-ILP rescue without a reliable positional mapping."
        )

    voxel = np.asarray(
        [1.625, 0.40625, 0.40625],
        dtype=np.float32,
    )

    n_nodes = len(coords)
    incoming_count = [0] * n_nodes
    outgoing: dict[int, list[tuple[int, float, float]]] = {}
    edge_set: set[tuple[int, int]] = set()

    # Fast frame-local candidate lookup. This keeps the post-ILP pass cheap:
    # each parent only scans detections in t+1 rather than every node.
    nodes_by_t: dict[int, list[int]] = {}
    for node_idx in range(n_nodes):
        nodes_by_t.setdefault(int(coords[node_idx, 0]), []).append(node_idx)

    edge_table = graph.edge_attrs()
    selected_edge_count = 0

    for row in edge_table.iter_rows(named=True):
        src_internal = int(row["source_id"])
        tgt_internal = int(row["target_id"])

        if (
            src_internal not in internal_to_global
            or tgt_internal not in internal_to_global
        ):
            continue

        src = internal_to_global[src_internal]
        tgt = internal_to_global[tgt_internal]

        prob = float(row.get("edge_prob", 1.0))

        delta = (
            coords[src, 1:].astype(np.float32)
            - coords[tgt, 1:].astype(np.float32)
        ) * voxel
        dist = float(np.linalg.norm(delta))

        outgoing.setdefault(src, []).append((tgt, prob, dist))
        incoming_count[tgt] += 1
        edge_set.add((src, tgt))
        selected_edge_count += 1

    if selected_edge_count == 0:
        return graph, []

    # Preserve V6's global safety cap.  In practice the biological gates below
    # should be much more restrictive than this cap.
    cap = max(
        1,
        int(round(selected_edge_count * max_division_fraction)),
    )

    proposals: list[dict] = []

    for parent, children in outgoing.items():
        if debug_parent is not None and parent == debug_parent:
            print(
                f"[V15 TARGET STATE] parent={parent} children={len(children)} "
                f"incoming={incoming_count[parent]} "
                f"division_prob={float(division_head_by_global.get(parent, 0.0)):.3f}",
                flush=True,
            )

        if len(children) != 1:
            if debug_parent is not None and parent == debug_parent:
                print("[V15 TARGET REJECT] parent does not have exactly one child", flush=True)
            continue
        if incoming_count[parent] <= 0:
            if debug_parent is not None and parent == debug_parent:
                print("[V15 TARGET REJECT] parent has no incoming track edge", flush=True)
            continue

        div_prob = float(division_head_by_global.get(parent, 0.0))
        if div_prob < division_head_threshold:
            if debug_parent is not None and parent == debug_parent:
                print(
                    f"[V15 TARGET REJECT] division_prob={div_prob:.3f} "
                    f"< {division_head_threshold:.3f}",
                    flush=True,
                )
            continue

        existing_child, existing_prob, _ = children[0]
        parent_t = int(coords[parent, 0])

        if int(coords[existing_child, 0]) != parent_t + 1:
            continue

        # Both branches must already show independent persistence to t+2 in
        # the final ILP graph. This is deliberately checked before scoring.
        succ_existing = outgoing.get(existing_child, [])
        if len(succ_existing) != 1:
            continue
        existing_next = succ_existing[0][0]
        if int(coords[existing_next, 0]) != parent_t + 2:
            continue

        p_xyz = coords[parent, 1:].astype(np.float32)
        e_xyz = coords[existing_child, 1:].astype(np.float32)

        existing_dist = float(
            np.linalg.norm((p_xyz - e_xyz) * voxel)
        )

        parent_candidates: list[dict] = []

        for candidate in nodes_by_t.get(parent_t + 1, []):
            if candidate == existing_child:
                continue
            if incoming_count[candidate] != 0:
                if debug_parent == parent and debug_candidate == candidate:
                    print(
                        f"[V15 TARGET CAND] candidate={candidate} rejected: "
                        f"incoming_count={incoming_count[candidate]}",
                        flush=True,
                    )
                continue
            if (parent, candidate) in edge_set:
                continue

            candidate_edge_prob = float(
                raw_edge_prob_by_pair.get((parent, candidate), 0.0)
            )
            if candidate_edge_prob < min_candidate_edge_prob:
                if debug_parent == parent and debug_candidate == candidate:
                    print(
                        f"[V15 TARGET CAND] candidate={candidate} rejected: "
                        f"raw_edge_prob={candidate_edge_prob:.3f} "
                        f"< {min_candidate_edge_prob:.3f}",
                        flush=True,
                    )
                continue

            succ_candidate = outgoing.get(candidate, [])
            if len(succ_candidate) != 1:
                if debug_parent == parent and debug_candidate == candidate:
                    print(
                        f"[V15 TARGET CAND] candidate={candidate} rejected: "
                        f"outgoing_count={len(succ_candidate)}",
                        flush=True,
                    )
                continue

            candidate_next = succ_candidate[0][0]
            if int(coords[candidate_next, 0]) != parent_t + 2:
                continue
            if candidate_next == existing_next:
                continue

            c_xyz = coords[candidate, 1:].astype(np.float32)

            cand_dist = float(
                np.linalg.norm((p_xyz - c_xyz) * voxel)
            )
            sister_dist = float(
                np.linalg.norm((e_xyz - c_xyz) * voxel)
            )

            midpoint = (e_xyz + c_xyz) / 2.0
            midpoint_error = float(
                np.linalg.norm((midpoint - p_xyz) * voxel)
            )

            v1 = (e_xyz - p_xyz) * voxel
            v2 = (c_xyz - p_xyz) * voxel
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom <= 0.0:
                continue

            angle_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(v1, v2) / denom,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )

            pair_score = score_mitosis_pair(
                edge1=existing_prob,
                edge2=candidate_edge_prob,
                dist1=existing_dist,
                dist2=cand_dist,
                sister_dist=sister_dist,
                midpoint_error=midpoint_error,
                angle_deg=angle_deg,
            )

            if debug_parent == parent and debug_candidate == candidate:
                print(
                    f"[V15 TARGET CAND] candidate={candidate} "
                    f"raw_edge_prob={candidate_edge_prob:.3f} "
                    f"pair_score={pair_score:.3f} "
                    f"d1={existing_dist:.2f} d2={cand_dist:.2f} "
                    f"sister={sister_dist:.2f} mid={midpoint_error:.2f} "
                    f"angle={angle_deg:.1f}",
                    flush=True,
                )

            if pair_score < pair_score_threshold:
                continue

            parent_candidates.append(
                {
                    "score": float(pair_score),
                    "candidate_edge_prob": candidate_edge_prob,
                    "division_prob": div_prob,
                    "parent": parent,
                    "existing_child": existing_child,
                    "candidate": candidate,
                    "cand_dist": cand_dist,
                    "sister_dist": sister_dist,
                    "midpoint_error": midpoint_error,
                    "angle_deg": angle_deg,
                }
            )

        if not parent_candidates:
            continue

        parent_candidates.sort(
            key=lambda x: (
                x["score"],
                x["candidate_edge_prob"],
                x["division_prob"],
            ),
            reverse=True,
        )

        best = parent_candidates[0]
        if len(parent_candidates) > 1:
            margin = best["score"] - parent_candidates[1]["score"]
            if margin < min_pair_margin:
                continue

        proposals.append(best)

    # After the learned biological gate, rank globally by the model's direct
    # parent->candidate edge probability. This stops pure geometry from
    # monopolising the sparse global division cap.
    proposals.sort(
        key=lambda x: (
            x["candidate_edge_prob"],
            x["score"],
            x["division_prob"],
        ),
        reverse=True,
    )

    if debug_parent is not None:
        target_rank = next(
            (rank for rank, p in enumerate(proposals, start=1)
             if int(p["parent"]) == debug_parent),
            None,
        )
        print(
            f"[V15 TARGET RANK] eligible_proposals={len(proposals)} "
            f"cap={cap} target_rank={target_rank}",
            flush=True,
        )

    additions: list[dict] = []
    used_parents: set[int] = set()
    used_children: set[int] = set()

    for prop in proposals:
        if len(additions) >= cap:
            break

        parent = int(prop["parent"])
        candidate = int(prop["candidate"])

        if parent in used_parents or candidate in used_children:
            continue
        if incoming_count[candidate] != 0:
            continue
        if (parent, candidate) in edge_set:
            continue

        additions.append(prop)
        used_parents.add(parent)
        used_children.add(candidate)
        incoming_count[candidate] = 1
        edge_set.add((parent, candidate))

    if not additions:
        return graph, []

    # No solver is run after this point: accepted family-completion edges are
    # the final topology written to GEFF. ``edge_prob`` stores the learned
    # biological confidence for traceability; evaluation uses graph topology.
    edge_columns = set(graph.edge_attrs().columns)
    if "edge_prob" not in edge_columns:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    if "edge_dist" not in edge_columns:
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)

    graph.bulk_add_edges(
        [
            {
                "source_id": positional_node_ids[int(prop["parent"])],
                "target_id": positional_node_ids[int(prop["candidate"])],
                "edge_prob": float(prop["score"]),
                "edge_dist": float(prop["cand_dist"]),
            }
            for prop in additions
        ]
    )

    return graph, additions


# =============================================================================
# Prediction loop
# =============================================================================

def predict(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_path: Path,
    cfg: PredictConfig,
    method: str = DEFAULT_METHOD,
    debug_video: Path | None = None,
    unet_batch_size: int = 4,
    video_slice: slice | None = None,
    evaluate: bool = False,
) -> None:
    """Run inference on the test split and save predictions as .geff files."""
    if debug_video is not None:
        # debug_video is a concrete dataset path.
        # Keep the actual path for loading, but use its stem for output naming.
        debug_path = debug_video
        data_dir = debug_path.parent

        if debug_path.name.endswith(".geff"):
            debug_name = debug_path.name[:-5]
        elif debug_path.name.endswith(".zarr"):
            debug_name = debug_path.name[:-5]
        else:
            debug_name = debug_path.name

        test_names = [debug_name]

    else:
        if splits_file.exists():
            folds = json.loads(splits_file.read_text())
        else:
            import random

            stems = sorted(
                p.stem
                for p in data_dir.glob("*.zarr")
                if (data_dir / f"{p.stem}.geff").exists()
            )

            random.Random(0).shuffle(stems)
            n_val = max(1, len(stems) // 10)

            folds = [{
                "train": stems[n_val:],
                "test": stems[:n_val],
            }]

        test_names = folds[fold]["test"]
        if video_slice is not None:
            test_names = test_names[video_slice]
    from dataspec import PREDICTIONS_PATH
    output_dir = PREDICTIONS_PATH / USERNAME / method / f"split_{fold}"
    if output_dir.exists():
        import shutil
        for old in output_dir.glob("*.geff"):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_size, downsample = load_model(weights_path, device)
    print(
        f"Fold {fold}: {len(test_names)} datasets | "
        f"weights={weights_path} | device={device} | window_size={window_size} | pool_kernel_um={cfg.pool_kernel_um}",
        flush=True,
    )

    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):
        if debug_video is not None:
            ds_path = debug_path
        else:
            ds_path = data_dir / name
        coords, edges, division_head_by_global, raw_edge_prob_by_pair = predict_video(
                model, ds_path, device,
                cfg=cfg,
                window_size=window_size,
                unet_batch_size=unet_batch_size,
                downsample=downsample,
            )
        graph, positional_node_ids = build_graph(coords, edges)
    
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)

        # V14 correction: perform the calibrated biological family-completion
        # rescue only after the final ILP topology is known.
        graph, post_ilp_rescues = add_post_ilp_biological_rescue(
            graph=graph,
            coords=coords,
            positional_node_ids=positional_node_ids,
            division_head_by_global=division_head_by_global,
            raw_edge_prob_by_pair=raw_edge_prob_by_pair,
            pair_score_threshold=0.20,
            division_head_threshold=0.30,
            min_candidate_edge_prob=cfg.threshold,
            min_pair_margin=0.0,
            max_division_fraction=0.00375,
            debug_parent=(668 if debug_video is not None and name == "6bba_268e1230" else None),
            debug_candidate=(703 if debug_video is not None and name == "6bba_268e1230" else None),
        )

        if post_ilp_rescues:
            print(
                f"  V15 post-ILP rescues added for {name}: "
                f"{len(post_ilp_rescues)}",
                flush=True,
            )
            for prop in post_ilp_rescues[:5]:
                print(
                    "    "
                    f"parent={int(prop['parent'])} "
                    f"existing={int(prop['existing_child'])} "
                    f"rescued={int(prop['candidate'])} "
                    f"pair_score={float(prop['score']):.3f} "
                    f"edge2={float(prop['candidate_edge_prob']):.3f} "
                    f"division_prob={float(prop['division_prob']):.3f}",
                    flush=True,
                )

        # One targeted, non-invasive diagnostic for the verified GT division
        # used during V13/V14 development. This makes the one-video test
        # decisive without changing any topology or thresholds.
        if debug_video is not None and name == "6bba_268e1230":
            target_rescue = next(
                (
                    prop for prop in post_ilp_rescues
                    if int(prop["parent"]) == 668
                ),
                None,
            )
            if target_rescue is None:
                print(
                    "[V15 TARGET] parent=668 was not rescued",
                    flush=True,
                )
            else:
                print(
                    "[V15 TARGET] "
                    f"parent=668 "
                    f"existing={int(target_rescue['existing_child'])} "
                    f"rescued={int(target_rescue['candidate'])} "
                    f"pair_score={float(target_rescue['score']):.3f} "
                    f"division_prob={float(target_rescue['division_prob']):.3f}",
                    flush=True,
                )

        save_graph(graph, output_dir / f"{name}.geff")

    print(f"Saved {len(test_names)} predictions to {output_dir}", flush=True)

    if evaluate:
        run = {
            "username": USERNAME,
            "method": method,
            "split": f"split_{fold}",
            "dir": output_dir,
            "geffs": sorted(output_dir.glob("*.geff")),
        }
        results = evaluate_run(run, gt_dir=data_dir)
        s = summarise(results)
        print(
            f"Evaluation ({len(results)} videos): "
            f"score={s['score']:.4f}  "
            f"edge_jaccard={s['edge_jaccard']:.4f}  "
            f"adj_edge_jaccard={s['adj_edge_jaccard']:.4f} (n_adj={s['n_adj']})  "
            f"division_jaccard={s['division_jaccard']:.4f} "
            f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})  "
            f"node_recall={s['node_recall']:.4f}  (n={s['n']})",
            flush=True,
        )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UNet + transformer edge prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Default: DATASET_PATH")
    parser.add_argument("--splits", type=str, default=None,
                        help="Default: DATASET_PATH/dataset_splits.json")
    parser.add_argument("--split", type=str, default="0",
                        help="Split index (0-4) or 'all'.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")
    parser.add_argument("--debug-video", type=str, default=None,
                        help="Path to a single dataset. Ignores fold/splits.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Python slice of the test list, e.g. ':1' for first video, "
                             "'2:5' for videos 2-4.")
    parser.add_argument("--unet-batch-size", type=int, default=4,
                        help="Number of frame pairs per UNet forward pass (default: 4).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against GT after saving predictions.")
    parser.add_argument("--det-threshold", type=float, default=0.96875,
                        help="Detection threshold for the V1-anchored hybrid.")
    parser.add_argument("--pool-kernel-um", type=float, default=5.0,
                        help="Physical local-max suppression distance in microns.")
    parser.add_argument("--no-det-tta", dest="det_tta", action="store_false", default=True,
                        help="Disable spatial detection TTA.")
    parser.add_argument("--edge-threshold",type=float,default=0.5,
                        help="Minimum sigmoid edge probability to keep as a candidate.")
    parser.add_argument("--use-ilp", action="store_true",
                        help="Post-process the predicted graph with the tracksdata ILP "
                             "solver (global, flow-consistent linking) instead of greedy "
                             "assignment. Needs pyscipopt; produces cleaner tracks.")
    parser.add_argument("--ilp-edge-weight", type=float, default=-1.0,
                        help="ILP: weight on edge_prob (default -1.0).")
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track appearing (default 0.1).")
    parser.add_argument("--ilp-disappearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track disappearing (default 0.1).")
    parser.add_argument("--ilp-division-weight", type=float, default=1.0,
                        help="ILP: cost of a division; lower to allow more splits (default 1.0).")

    args = parser.parse_args()

    from dataspec import DATASET_PATH
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    splits_file = Path(args.splits) if args.splits else data_dir / "dataset_splits.json"
    debug_video = Path(args.debug_video) if args.debug_video else None
    video_slice = (
        slice(*[int(x) if x else None for x in args.slice.split(":")])
        if args.slice else None
    )
    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        det_tta=args.det_tta,
        pool_kernel_um=args.pool_kernel_um,
        threshold=args.edge_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )

    folds = range(5) if args.split == "all" else [int(args.split)]

    for fold in folds:
        weights_path = (
            Path(args.weights) if args.weights
            else WEIGHTS_PATH / args.method / f"split_{fold}" / "edge_predictor_best.pth"
        )
        predict(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_path=weights_path,
            cfg=cfg,
            method=args.method,
            debug_video=debug_video,
            unet_batch_size=args.unet_batch_size,
            video_slice=video_slice,
            evaluate=args.evaluate,
        )


if __name__ == "__main__":
    main()
