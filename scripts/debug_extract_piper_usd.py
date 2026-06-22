"""Inspect Piper USD joints/links for cuRobo config generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path("atec_robot_model/robot/_piper/piper_flattened.usd"),
    )
    parser.add_argument(
        "--joints-only",
        action="store_true",
        help="Only print USD joint metadata and skip the full prim listing.",
    )
    parser.add_argument(
        "--no-close",
        action="store_true",
        help="Flush stdout and exit without SimulationApp.close(). Useful around Isaac close-time crashes.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {args.usd}")

    default_prim = stage.GetDefaultPrim()
    print(f"stage={args.usd}")
    print(f"default_prim={default_prim.GetPath() if default_prim else None}")

    if not args.joints_only:
        print("\n[Prims]")
        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if type_name or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps() if xform else []
                op_values = []
                for op in ops:
                    op_values.append((op.GetOpName(), op.Get()))
                print(
                    f"{prim.GetPath()} type={type_name} "
                    f"rigid={prim.HasAPI(UsdPhysics.RigidBodyAPI)} ops={op_values}"
                )

    print("\n[Joints]")
    count = 0
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if "Joint" not in type_name and not prim.HasAPI(UsdPhysics.Joint):
            continue
        count += 1
        print(f"JOINT {prim.GetPath()} type={type_name}")
        for rel in prim.GetRelationships():
            print(f"  rel {rel.GetName()}: {[str(p) for p in rel.GetTargets()]}")
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if (
                name.startswith("physics:")
                or name.startswith("drive:")
                or name.startswith("limit:")
                or name.startswith("state:")
                or name in {"xformOp:translate", "xformOp:orient", "xformOp:transform"}
            ):
                print(f"  {name}: {attr.Get()}")
    print(f"joint_count={count}")
    sys.stdout.flush()

    if args.no_close:
        os._exit(0)

    simulation_app.close()


if __name__ == "__main__":
    main()
