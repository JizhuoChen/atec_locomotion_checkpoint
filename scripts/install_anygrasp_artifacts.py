#!/usr/bin/env python3
"""Install AnyGrasp license/checkpoint artifacts into the local SDK layout."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANYGRASP_ROOT = REPO_ROOT / "third_party/anygrasp_sdk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anygrasp-root", type=Path, default=DEFAULT_ANYGRASP_ROOT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--license-dir", type=Path, help="Extracted AnyGrasp license directory.")
    group.add_argument("--license-zip", type=Path, help="Returned AnyGrasp license zip.")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint_detection.tar path.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy artifacts instead of creating symlinks.",
    )
    return parser.parse_args()


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    src = src.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_existing(dst)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def extract_license_zip(path: Path, target_root: Path) -> Path:
    extract_dir = target_root / "_license_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path.expanduser().resolve()) as archive:
        archive.extractall(extract_dir)
    candidates = [p.parent for p in extract_dir.rglob("licenseCfg.json")]
    if not candidates:
        raise FileNotFoundError(f"No licenseCfg.json found inside {path}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    anygrasp_root = args.anygrasp_root.expanduser().resolve()
    detection_dir = anygrasp_root / "grasp_detection"
    tracking_dir = anygrasp_root / "grasp_tracking"
    if not detection_dir.exists():
        raise FileNotFoundError(f"Missing AnyGrasp detection directory: {detection_dir}")

    installed = {}
    if args.license_zip is not None:
        license_dir = extract_license_zip(args.license_zip, anygrasp_root)
    else:
        license_dir = args.license_dir.expanduser().resolve() if args.license_dir else None

    if license_dir is not None:
        if not (license_dir / "licenseCfg.json").exists():
            raise FileNotFoundError(f"License directory lacks licenseCfg.json: {license_dir}")
        link_or_copy(license_dir, detection_dir / "license", args.copy)
        installed["detection_license"] = str((detection_dir / "license").resolve())
        if tracking_dir.exists():
            link_or_copy(license_dir, tracking_dir / "license", args.copy)
            installed["tracking_license"] = str((tracking_dir / "license").resolve())

    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        link_or_copy(checkpoint, detection_dir / "log/checkpoint_detection.tar", args.copy)
        installed["checkpoint"] = str((detection_dir / "log/checkpoint_detection.tar").resolve())

    if not installed:
        raise SystemExit("Nothing installed. Provide --license-dir/--license-zip and/or --checkpoint.")

    for name, path in installed.items():
        print(f"[INFO] {name}: {path}")


if __name__ == "__main__":
    main()
