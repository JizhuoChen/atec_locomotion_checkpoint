#!/usr/bin/env python3
"""Verify the code and artifacts for the canonical August 7 model_19998 run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

EXPECTED = {
    "scripts/rsl_rl/continue_b2_piper_rough.py": "4d6357c26d67fc2dbbf3adb53541c31773aa5d9e7404246056902a7f4b97d4b9",
    "scripts/rsl_rl/train.py": "9ea2271a1811211d158e275c5b7ec687972d718c25a6393bb08a0fd74938b1be",
    "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/robust_terrain_cfg.py": "8f5d5e7101f0bcae146ddfa9cfd2fcfbbd3de741645fd29a01bd9eac73cb71f9",
    "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/robust_piper_env_cfg.py": "69231870ee6ea86a1f411e9887176ad3dd48f81e330a2571f00820b1e45bcaf4",
    "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/commands.py": "8c8e94e2e302262777f9a9c76fafd5e7567ed763573d174f9f643cb695409f9c",
    "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/robust_events.py": "805c58d27c573b09a2a8d6dccc002889a821be33c16a40524fe0a5ea138450dd",
    "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/piper_env_cfg.py": "68902e1e7929451db621339fa7ccff96276914960b5351fda4def46020af4969",
    "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/curriculums.py": "a2d766a477b7b5f64729acbea118a0e37bee332ce8eb43dadce81742ab15bbf7",
    "logs/rsl_rl/unitree_b2_piper_robust_heading_rough/2026-08-06_23-17-23_b2_piper_robust_20260806_215402_69aaacaf_heading_rough/model_11999.pt": "c7a8b3b0e990c7fb724ea4408c0c2decfee8cfec5e7c0d9938d88bfa416813b8",
    "reference/lineage/flat/model_7999.pt": "8d9d184d94cf707c45550c715ef5486547976cde2149d879cba9cf6aaedd8264",
    "reference/lineage/heading_rough/model_11999.pt": "c7a8b3b0e990c7fb724ea4408c0c2decfee8cfec5e7c0d9938d88bfa416813b8",
    "reference/model_19998_run/model_19998.pt": "995b9d11ae99648255e5c213baaa14cfbe31b0380f76e976a3469a593322bfd9",
    "reference/model_19998_run/exported/policy.pt": "c2499e63771e739ff61c2e9a2acd1d6a045da2094f5f345a9ff35a3c67731d40",
    "reference/model_19998_run/exported/policy.onnx": "688bdbc4897c6dd9cc1cce54249131a35e9206614cfad2f383d2e907978b67eb",
    "reference/model_19998_run/params/env.yaml": "da74d7e5828dd368fe1625113ff90497387f550cfa17574b5b7d213555cc13ec",
    "reference/model_19998_run/params/agent.yaml": "ec6d2670f17d5c776f5ae98cfc7c603c94d73dfe588f159d8a7e8a9719d08b13",
    "reference/model_19998_run/params/resume.yaml": "5dfecd49c29abe6b72b9628accef466ba3ad98697cfba14145cdde4b0b23fe6b",
    "reference/model_19998_run/params/spawn_audit.yaml": "2b71d517e084f062733af849c5e7fcfd3a8352385e5c324b0374f8421bac3f30",
    "reference/provenance/pipeline_aug07_model_19998.json": "45f07d2ef295156f8d6ccaddc693c16348436ab0237790bfb3b49418153af49c",
    "atec_robot_model/robot/b2/b2_piper.usda": "83e17a50c68f6aa184815bbe50c81155613f5b96844aa76d0d888b46a4741222",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_all_assets() -> list[str]:
    failures = []
    manifest = ROOT / "reference/provenance/atec_robot_model.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"MISSING {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"HASH {relative}: expected {expected}, got {actual}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-assets",
        action="store_true",
        help="Also hash every file in the 635 MB robot asset tree.",
    )
    args = parser.parse_args()

    failures = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"MISSING {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"HASH {relative}: expected {expected}, got {actual}")

    if args.all_assets:
        failures.extend(verify_all_assets())

    base = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "2f4bd998386717bd8e4484db43fc8e3b9c0aee5c",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if base.returncode != 0:
        failures.append("Expected base commit 2f4bd998386... is not an ancestor of HEAD")

    if failures:
        print("Snapshot verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    qualifier = "including all robot assets" if args.all_assets else "critical files"
    print(f"Snapshot verification passed ({qualifier}).")
    print("Canonical checkpoint SHA-256: 995b9d11ae99648255e5c213baaa14cfbe31b0380f76e976a3469a593322bfd9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
