#!/usr/bin/env python3
"""Evaluate an existing grasp directory with the Task E CuRobo gate."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object-name", default="mustard_bottle")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--task", default="ATEC-TaskE-Piper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grasp_dir = args.grasp_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_output = output_dir / "_pipeline_context"
    pipeline_argv = [
        str(SCRIPT_DIR / "task_e_full_anygrasp_ee_pipeline.py"),
        "--seed",
        str(args.seed),
        "--output",
        str(pipeline_output),
        "--object-order",
        args.object_name,
        "--grasp-selection",
        "curobo_first_ik",
        "--ik-filter-top-k",
        str(args.top_k),
        "--device",
        args.device,
    ]
    if args.headless:
        pipeline_argv.append("--headless")
    if args.disable_fabric:
        pipeline_argv.append("--disable_fabric")

    old_argv = sys.argv[:]
    sys.argv = pipeline_argv
    try:
        pipeline = importlib.import_module("task_e_full_anygrasp_ee_pipeline")
    finally:
        sys.argv = old_argv

    env_cfg = pipeline.parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: WPS433

    env = pipeline.gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]

    result_path = grasp_dir / "anygrasp_result.json"
    if not result_path.exists():
        result_path = grasp_dir / "graspgen_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    top_grasps_path = grasp_dir / "top_grasps.json"
    if top_grasps_path.exists():
        result.setdefault("anygrasp", {})["status"] = "ok"
        result.setdefault("anygrasp", {})["top_grasps"] = json.loads(top_grasps_path.read_text(encoding="utf-8"))
    final_pose_path = grasp_dir / "final_grasp_pose.json"
    if not final_pose_path.exists():
        top = json.loads((grasp_dir / "top_grasps.json").read_text(encoding="utf-8"))
        if top:
            final_pose_path.write_text(json.dumps(top[0], indent=2), encoding="utf-8")

    record = {
        "anygrasp_result_path": str(result_path),
        "final_grasp_pose_path": str(final_pose_path),
        "anygrasp_result": result,
    }
    records = {args.object_name: record}
    object_poses = copy.deepcopy(pipeline.deterministic_object_poses(args.seed))
    overlay_center = (result.get("top_grasps_overlay_geometry") or {}).get("object_center_world")
    if overlay_center is not None:
        object_poses[args.object_name]["center_w"] = [
            float(overlay_center[0]),
            float(overlay_center[1]),
            float(pipeline.TABLE_TOP_Z + 0.05),
        ]
        object_poses[args.object_name]["source"] = "existing_grasp_dir_overlay_center_xy"

    try:
        pipeline.select_curobo_feasible_grasps(env, obs, robot, object_poses, records)
    finally:
        env.close()
        try:
            pipeline.simulation_app.close()
        except Exception:
            pass

    selection_path = grasp_dir / "ik_candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = selection.get("candidates", [])
    summary = {
        "grasp_dir": str(grasp_dir),
        "selection_path": str(selection_path),
        "mode": selection.get("mode"),
        "top_k": selection.get("top_k"),
        "candidate_count": len(candidates),
        "geometry_pass_count": sum(not item.get("skipped_curobo", False) for item in candidates),
        "all_ik_success_count": sum(bool(item.get("all_ik_success", False)) for item in candidates),
        "accepted_by_policy_count": sum(bool(item.get("accepted_by_policy", False)) for item in candidates),
        "strict_ok_count": sum(bool(item.get("ok", False)) for item in candidates),
        "skipped_final_geometry_count": sum(item.get("skip_reason") == "final_piper_geometry" for item in candidates),
        "skipped_approach_count": sum(item.get("skip_reason") == "approach_piper_geometry" for item in candidates),
        "accepted_ranks": [item.get("rank") for item in candidates if item.get("accepted_by_policy")],
        "all_ik_success_ranks": [item.get("rank") for item in candidates if item.get("all_ik_success")],
        "strict_ok_ranks": [item.get("rank") for item in candidates if item.get("ok")],
        "selected_rank": selection.get("selected_rank"),
        "selected_ok": selection.get("selected_ok"),
        "selected_strict_ok": selection.get("selected_strict_ok"),
        "fallback_used": selection.get("fallback_used"),
    }
    summary_path = output_dir / "curobo_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
