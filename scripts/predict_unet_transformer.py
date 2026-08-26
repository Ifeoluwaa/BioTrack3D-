#!/usr/bin/env python
"""Run UNet + transformer edge prediction on datasets and export to .geff.

Usage:
    uv run scripts/predict_unet_transformer.py --split 0
"""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
from train_unet_transformer import (
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
) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detection coords and predicted edges.

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

    return graph


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
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once. UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
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

            if t_src == 24 and t_tgt == 25:
                order = np.argsort(
                    division_probs
                )[::-1]

                print(
                    "\n=== DIVISION HEAD DIAGNOSTIC t=24->25 ===",
                    flush=True,
                )

                for rank, i in enumerate(
                    order[:10],
                    start=1,
                ):
                    coord = (
                        p_coords_src[0, i] * ds_arr_t
                    ).detach().cpu().numpy()

                    global_id = int(idx_src[i])

                    print(
                        f"{rank}. "
                        f"local={i} "
                        f"global={global_id} "
                        f"coord={coord.tolist()} "
                        f"division_prob="
                        f"{float(division_probs[i]):.6f}",
                        flush=True,
                    )

            # Temporary division-head diagnostic.
            # Trigger on the actual source frame coordinates, not the
            # local window index.
            # ----------------------------------------------------------
            if (
                p_coords_src.shape[1] > 0
                and float(
                    p_coords_src[0, :, 0].min().detach().cpu()
                ) <= 45.0
                <= float(
                    p_coords_src[0, :, 0].max().detach().cpu()
                )
            ):
                order = np.argsort(
                    division_probs
                )[::-1]

                print(
                    "\n=== DIVISION HEAD DIAGNOSTIC ===",
                    flush=True,
                )

                for rank, i in enumerate(
                    order[:10],
                    start=1,
                ):
                    coord = (
                        p_coords_src[0, i] * ds_arr_t
                    ).detach().cpu().numpy()

                    global_id = int(idx_src[i])

                    print(
                        f"{rank}. "
                        f"local={i} "
                        f"global={global_id} "
                        f"coord={coord.tolist()} "
                        f"division_prob="
                        f"{float(division_probs[i]):.6f}",
                        flush=True,
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

                    if prob <= cfg.threshold:
                        continue

                    # Original-resolution voxel displacement.
                    delta_voxel = (
                        src_xyz[i]
                        - tgt_xyz[j]
                    )

                    # Convert each spatial axis to physical units.
                    physical_delta = (
                        delta_voxel
                        * np.asarray(
                            [1.625, 0.40625, 0.40625],
                            dtype=np.float32,
                        )
                    )

                    dist = float(
                        np.linalg.norm(
                            physical_delta
                        )
                    )

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
            # Build joint division hypotheses.
            #
            # A division hypothesis is a parent + two daughters selected
            # together. This is deliberately done BEFORE greedy edge
            # selection so the two daughter edges can be reserved as a pair.
            # ==========================================================
            division_pairs: list[dict] = []

            by_parent: dict[int, list[dict]] = {}

            for cand in candidate_data:
                if cand["prob"] >= MITOSIS_MIN_EDGE_PROB:
                    by_parent.setdefault(
                        cand["i"],
                        [],
                    ).append(cand)

            for parent_idx, parent_candidates in by_parent.items():

                if len(parent_candidates) < 2:
                    continue

                for a in range(len(parent_candidates)):
                    for b in range(a + 1, len(parent_candidates)):

                        ca = parent_candidates[a]
                        cb = parent_candidates[b]

                        # ------------------------------------------------------
                        # Parent -> daughter distances.
                        # ------------------------------------------------------
                        if ca["dist"] > MITOSIS_PARENT_MAX_UM:
                            continue

                        if cb["dist"] > MITOSIS_PARENT_MAX_UM:
                            continue

                        # ------------------------------------------------------
                        # Daughter-to-daughter distance.
                        # ------------------------------------------------------
                        daughter_delta_voxel = (
                            tgt_xyz[ca["j"]]
                            - tgt_xyz[cb["j"]]
                        )

                        physical_daughter_delta = (
                            daughter_delta_voxel
                            * np.asarray(
                                [1.625, 0.40625, 0.40625],
                                dtype=np.float32,
                            )
                        )

                        daughter_dist = float(
                            np.linalg.norm(
                                physical_daughter_delta
                            )
                        )

                        if not (
                            MITOSIS_DAUGHTER_MIN_UM
                            <= daughter_dist
                            <= MITOSIS_DAUGHTER_MAX_UM
                        ):
                            continue

                        # ------------------------------------------------------
                        # Symmetry of the two parent->daughter distances.
                        # ------------------------------------------------------
                        symmetry = (
                            abs(ca["dist"] - cb["dist"])
                            / max(
                                ca["dist"] + cb["dist"],
                                1e-6,
                            )
                        )

                        if symmetry > 0.75:
                            continue

                        # ------------------------------------------------------
                        # Joint pair score.
                        #
                        # This is intentionally pair-level rather than adding
                        # independent bonuses to the two edges.
                        # ------------------------------------------------------
                        pair_edge_score = 0.5 * (
                            ca["prob"] + cb["prob"]
                        )

                        pair_geometry_score = (
                            1.0 - min(symmetry, 1.0)
                        )

                        pair_score = (
                            0.8 * pair_edge_score
                            + 0.2 * pair_geometry_score
                            + MITOSIS_BONUS
                        )

                        division_pairs.append(
                            {
                                "parent": parent_idx,
                                "daughter1": ca["j"],
                                "daughter2": cb["j"],
                                "score": float(pair_score),
                                "prob1": float(ca["prob"]),
                                "prob2": float(cb["prob"]),
                                "dist1": float(ca["dist"]),
                                "dist2": float(cb["dist"]),
                                "daughter_dist": float(
                                    daughter_dist
                                ),
                                "symmetry": float(
                                    symmetry
                                ),
                            }
                        )

            division_pairs.sort(
                key=lambda x: x["score"],
                reverse=True,
            )
            # ==========================================================
            # Preserve original greedy association mechanism.
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

            # ==========================================================
            # Joint division selection.
            #
            # Reserve both daughter edges together before ordinary greedy
            # edges are considered.
            # ==========================================================
            selected_division_parents: set[int] = set()

            for pair in division_pairs:

                parent_idx = pair["parent"]
                daughter1_idx = pair["daughter1"]
                daughter2_idx = pair["daughter2"]

                # Parent must not already have a child.
                if children_count.get(parent_idx, 0) != 0:
                    continue

                # Neither daughter can already have a parent.
                if parents_count.get(daughter1_idx, 0) != 0:
                    continue

                if parents_count.get(daughter2_idx, 0) != 0:
                    continue

                gi = int(
                    idx_src[parent_idx]
                )

                gj1 = int(
                    idx_tgt[daughter1_idx]
                )

                gj2 = int(
                    idx_tgt[daughter2_idx]
                )

                # Add BOTH edges as one division hypothesis.
                all_edges.append(
                    (
                        gi,
                        gj1,
                        float(pair["prob1"]),
                        float(pair["dist1"]),
                    )
                )

                all_edges.append(
                    (
                        gi,
                        gj2,
                        float(pair["prob2"]),
                        float(pair["dist2"]),
                    )
                )

                children_count[parent_idx] = 2
                parents_count[daughter1_idx] = 1
                parents_count[daughter2_idx] = 1

                selected_division_parents.add(
                    parent_idx
                )

            
            # ----------------------------------------------------------
            # Greedy edge selection.
            # ----------------------------------------------------------
            for (
                score,
                prob,
                i,
                j,
                dist,
            ) in candidates:

                if i in selected_division_parents:
                    continue

                n_ch = children_count.get(
                    i,
                    0,
                )

                n_pa = parents_count.get(
                    j,
                    0,
                )

                if (
                    cfg.max_children_per_node
                    is not None
                    and n_ch
                    >= cfg.max_children_per_node
                ):

                    continue

                if (
                    cfg.max_parents_per_node
                    is not None
                    and n_pa
                    >= cfg.max_parents_per_node
                ):
                    continue

                gi = int(
                    idx_src[i]
                )

                gj = int(
                    idx_tgt[j]
                )

                all_edges.append(
                    (
                        gi,
                        gj,
                        float(prob),
                        float(dist),
                    )
                )


                children_count[i] = (
                    n_ch + 1
                )

                parents_count[j] = (
                    n_pa + 1
                )

        del unet_out

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

    return coords, all_edges
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
        coords, edges = predict_video(
                model, ds_path, device,
                cfg=cfg,
                window_size=window_size,
                unet_batch_size=unet_batch_size,
                downsample=downsample,
            )
        graph = build_graph(coords, edges)
    
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)
              
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
    parser.add_argument("--det-threshold", type=float, default=0.3,
                        help="Min raw logit for a detection peak to be kept. "
                             "Default 0.3: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
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
