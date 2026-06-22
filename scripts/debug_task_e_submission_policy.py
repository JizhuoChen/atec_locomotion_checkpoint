"""Run a Task E submission policy and print score/progress diagnostics.

This is intentionally separate from ``scripts/play_atec_task.py`` because that
script imports ``demo.solution``.  This runner imports a chosen submission
directory directly, which is useful for checking packaged submissions.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ATEC-TaskE-Piper")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3500)
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--submission-dir",
        default="submissions/task_e_act_baseline_root_submission",
        help="Directory containing solution.py and policy assets.",
    )
    parser.add_argument(
        "--heartbeat",
        type=int,
        default=250,
        help="Print a progress line every N simulator steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional environment seed for reproducible local submission rollouts.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def load_solution(submission_dir: Path):
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location(
        "task_e_submission_solution", submission_dir / "solution.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solution.py from {submission_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlgSolution()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: WPS433
    import torch  # noqa: WPS433
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: WPS433
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: WPS433

    import atec_rl_lab.tasks  # noqa: F401, WPS433

    repo_root = Path(__file__).resolve().parents[1]
    submission_dir = (repo_root / args.submission_dir).resolve()
    solution = load_solution(submission_dir)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    if args.seed is not None:
        env_cfg.seed = int(args.seed)
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()
    total_score = 0.0
    done_step = None
    start_time = time.time()

    with torch.inference_mode():
        for step in range(args.max_steps):
            resp = solution.predicts(obs, total_score)
            if resp.get("giveup"):
                print(f"GIVEUP step={step} score={total_score:.2f}", flush=True)
                break

            actions = torch.tensor(
                resp["action"], dtype=torch.float32, device=args.device
            ).view(args.num_envs, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            sim_dt = info.get("Step_dt", env.unwrapped.step_dt)
            sim_dt = sim_dt.item() if hasattr(sim_dt, "item") else float(sim_dt)
            score_inc = (
                reward.mean().item() / sim_dt
                if isinstance(reward, torch.Tensor)
                else float(reward) / sim_dt
            )
            if score_inc:
                total_score += score_inc
                print(
                    f"SCORE step={step} inc={score_inc:.2f} total={total_score:.2f}",
                    flush=True,
                )
            if args.heartbeat and step % args.heartbeat == 0:
                scripted_step = getattr(solution, "_scripted_step", None)
                scripted_len = len(getattr(solution, "_scripted_actions", []))
                scripted_started = getattr(solution, "_scripted_started", False)
                scripted_targets = getattr(solution, "_scripted_targets", [])
                suffix = (
                    f" scripted={scripted_started}:{scripted_step}/{scripted_len}"
                    if scripted_started or scripted_len
                    else ""
                )
                if scripted_targets and step % max(args.heartbeat, 1) == 0:
                    target_bits = [
                        f"{name}:{'none' if est is None else f'{est:.3f}'}->{used:.3f}"
                        for name, est, used in scripted_targets
                    ]
                    suffix += " targets=" + ",".join(target_bits)
                print(f"HEARTBEAT step={step} score={total_score:.2f}{suffix}", flush=True)

            is_done = bool(terminated.item() or truncated.item())
            if is_done:
                done_step = step
                print(
                    "DONE "
                    f"step={step} terminated={bool(terminated.item())} "
                    f"truncated={bool(truncated.item())} score={total_score:.2f}",
                    flush=True,
                )
                break
        else:
            print(f"MAX_STEPS step={args.max_steps} score={total_score:.2f}", flush=True)

    scene = env.unwrapped.scene
    origin = scene.env_origins[0]
    for name in ("object_1", "object_2", "object_3"):
        pos = scene[name].data.root_pos_w[0, :3] - origin
        rounded = [round(float(value), 4) for value in pos.tolist()]
        print(f"OBJ {name} pos_local={rounded}", flush=True)

    print(
        f"FINAL score={total_score:.2f} done_step={done_step} "
        f"wall_s={time.time() - start_time:.1f}",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
