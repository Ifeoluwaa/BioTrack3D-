#!/usr/bin/env python3
"""
BioTrack3D++ V16.6b FROZEN canonical predictor.

Frozen production stack
-----------------------
V16.4a  neural detector/edge tracker + learned mitosis biology
    -> association candidate cache + division audit
V16.5a  frozen tracklet-pairwise HGB association postprocessor
V16.6a  compact-family veto at 0.50
V16.6b  narrow established-like low-pair family override at 0.50

V16.7 is intentionally excluded.

This file is an orchestrator: it keeps the validated stage implementations and
frozen model artifacts separate rather than duplicating thousands of lines of
already-validated neural/association code.

Important environment split
---------------------------
The preserved V16.5a joblib requires scikit-learn 1.6.1 compatibility.
V16.6 family inference must run in the normal project environment. Therefore
only the V16.5a subprocess receives the optional compatibility PYTHONPATH.

For hidden test deployment, V16.6 uses the frozen FULL family-context model
trained on all labeled training families. Hidden test videos are not in that
training set. Local V16.6 research benchmarks used leave-current-video-out
models to avoid training-video contamination; do not expect this deployment
mode to reproduce those held-out family scores on the 19 labeled videos.

No V16.7. No threshold tuning.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, *, cwd, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        check=True,
    )


def require(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)

    ap.add_argument("--method", default="unet_transformer")
    ap.add_argument("--split", default="0")
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--debug-video", type=Path, default=None)
    ap.add_argument("--slice", dest="video_slice", default=None)

    ap.add_argument("--v16-5a-model", type=Path, required=True)
    ap.add_argument("--family-model", type=Path, required=True)
    ap.add_argument("--family-metadata", type=Path, required=True)

    ap.add_argument(
        "--sklearn161-path",
        type=Path,
        default=None,
        help=(
            "Directory containing the scikit-learn 1.6.1 compatibility "
            "installation used only to load/run the frozen V16.5a joblib. "
            "Example: .compat/sklearn161"
        ),
    )

    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/v16_6b_frozen_run"),
    )
    ap.add_argument(
        "--final-output-dir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--skip-v16-4a",
        action="store_true",
        help=(
            "Replay mode: do not rerun neural inference. Requires "
            "--base-pred-dir, --candidate-dir, and --division-audit."
        ),
    )
    ap.add_argument("--base-pred-dir", type=Path, default=None)
    ap.add_argument("--candidate-dir", type=Path, default=None)
    ap.add_argument("--division-audit", type=Path, default=None)

    args = ap.parse_args()

    repo = args.repo.resolve()
    data_dir = args.data_dir.resolve()
    weights = args.weights.resolve()

    scripts = repo / "scripts"
    src = repo / "src"

    v16_4a_script = scripts / "predict_biotrack3d_v16_4a_assoc_audit.py"
    v16_5a_script = scripts / "apply_v16_5a_pairwise_postprocess_fast.py"
    family_script = scripts / "apply_v16_6ab_family_postprocess_frozen.py"

    for p, label in (
        (v16_4a_script, "V16.4a association-audit predictor"),
        (v16_5a_script, "V16.5a fast postprocessor"),
        (family_script, "V16.6a/b family postprocessor"),
        (weights, "neural weights"),
        (args.v16_5a_model.resolve(), "V16.5a frozen model"),
        (args.family_model.resolve(), "V16.6 frozen family model"),
        (args.family_metadata.resolve(), "V16.6 family metadata"),
    ):
        require(p, label)

    work = (
        args.work_dir
        if args.work_dir.is_absolute()
        else repo / args.work_dir
    ).resolve()
    work.mkdir(parents=True, exist_ok=True)

    v16_4a_audit = (
        args.division_audit.resolve()
        if args.division_audit is not None
        else work / "v16_4a_division_audit.csv"
    )
    candidate_dir = (
        args.candidate_dir.resolve()
        if args.candidate_dir is not None
        else work / "v16_4a_assoc_candidates"
    )
    v16_5a_dir = work / "v16_5a_geff"

    final_dir = (
        args.final_output_dir.resolve()
        if args.final_output_dir is not None
        else work / "v16_6b_final_geff"
    )

    # ------------------------------------------------------------------
    # Stage 1: V16.4a neural inference + frozen mitosis + candidate cache.
    # ------------------------------------------------------------------
    if args.skip_v16_4a:
        if args.base_pred_dir is None:
            raise RuntimeError(
                "--skip-v16-4a requires --base-pred-dir."
            )
        if args.candidate_dir is None or args.division_audit is None:
            raise RuntimeError(
                "--skip-v16-4a requires --candidate-dir and --division-audit."
            )
        base_pred_dir = args.base_pred_dir.resolve()
        require(base_pred_dir, "replay V16.4a graph directory")
        require(candidate_dir, "replay association candidate directory")
        require(v16_4a_audit, "replay division audit")
        print("V16.4a neural stage: SKIPPED (replay mode)")
    else:
        candidate_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            v16_4a_script,
            "--method", args.method,
            "--data-dir", data_dir,
            "--split", args.split,
            "--weights", weights,
            "--audit-csv", v16_4a_audit,
            "--association-candidates-dir", candidate_dir,
        ]
        if args.splits is not None:
            cmd += ["--splits", args.splits.resolve()]
        if args.debug_video is not None:
            cmd += ["--debug-video", args.debug_video.resolve()]
        if args.video_slice is not None:
            cmd += ["--slice", args.video_slice]

        run(cmd, cwd=repo)

        # Derive the exact standard output path from project dataspec.
        sys.path.insert(0, str(src))
        sys.path.insert(0, str(scripts))
        from dataspec import USERNAME, PREDICTIONS_PATH
        base_pred_dir = (
            Path(PREDICTIONS_PATH)
            / USERNAME
            / args.method
            / f"split_{int(args.split)}"
        ).resolve()

        require(base_pred_dir, "V16.4a prediction directory")
        require(v16_4a_audit, "V16.4a division audit")
        require(candidate_dir, "V16.4a association candidates")

    # ------------------------------------------------------------------
    # Stage 2: exact frozen V16.5a association.
    # ------------------------------------------------------------------
    env = os.environ.copy()
    if args.sklearn161_path is not None:
        compat = args.sklearn161_path.resolve()
        require(compat, "scikit-learn 1.6.1 compatibility directory")
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(compat) if not old else str(compat) + os.pathsep + old
        )

    if v16_5a_dir.exists():
        shutil.rmtree(v16_5a_dir)

    run(
        [
            sys.executable,
            v16_5a_script,
            "--repo", repo,
            "--data-dir", data_dir,
            "--pred-dir", base_pred_dir,
            "--candidate-dir", candidate_dir,
            "--division-audit", v16_4a_audit,
            "--model", args.v16_5a_model.resolve(),
            "--output-dir", v16_5a_dir,
        ],
        cwd=repo,
        env=env,
    )

    # ------------------------------------------------------------------
    # Stage 3+4: frozen V16.6a and V16.6b family context in NORMAL env.
    # ------------------------------------------------------------------
    if final_dir.exists():
        shutil.rmtree(final_dir)

    run(
        [
            sys.executable,
            family_script,
            "--repo", repo,
            "--data-dir", data_dir,
            "--raw-v16-4a-pred-dir", base_pred_dir,
            "--v16-5a-pred-dir", v16_5a_dir,
            "--division-audit", v16_4a_audit,
            "--family-model", args.family_model.resolve(),
            "--family-metadata", args.family_metadata.resolve(),
            "--output-dir", final_dir,
            "--actions-out", work / "v16_6ab_family_actions.csv",
        ],
        cwd=repo,
        env=os.environ.copy(),
    )

    n_base = len(list(base_pred_dir.glob("*.geff")))
    n_cache = len(list(candidate_dir.glob("*.npz")))
    n_v5 = len(list(v16_5a_dir.glob("*.geff")))
    n_final = len(list(final_dir.glob("*.geff")))

    if not (n_base == n_cache == n_v5 == n_final and n_final > 0):
        raise RuntimeError(
            "Stage video-count mismatch: "
            f"V16.4a={n_base}, cache={n_cache}, "
            f"V16.5a={n_v5}, final={n_final}"
        )

    print("\n=== V16.6b FROZEN PREDICTION COMPLETE ===")
    print(f"videos:              {n_final}")
    print(f"V16.4a graphs:       {base_pred_dir}")
    print(f"candidate cache:      {candidate_dir}")
    print(f"division audit:       {v16_4a_audit}")
    print(f"V16.5a graphs:        {v16_5a_dir}")
    print(f"FINAL V16.6b graphs:  {final_dir}")
    print("V16.7 included:       NO")


if __name__ == "__main__":
    main()
