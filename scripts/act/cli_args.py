"""CLI argument definitions for Task E demo collection.
"""


def add_collect_demo_args(parser) -> None:
    parser.add_argument(
        "--num_demos", type=int, default=50,
        help="Number of successful demos to collect.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./datasets/atec_task_e",
        help="Directory to save trajectory.hdf5 and trajectory.json.",
    )
    parser.add_argument(
        "--pick_objects", type=int, nargs="+", default=[1, 2, 3],
        metavar="N",
        help="Which objects to pick (subset of {1,2,3}). Default: all three.",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False,
        help="Save an MP4 per demo for visualization "
             "(requires: pip install imageio imageio-ffmpeg).",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Output directory for MP4 files. Defaults to <output_dir>/videos/.",
    )
    parser.add_argument(
        "--save_images", action="store_true", default=False,
        help="Save raw RGB frames into HDF5 under traj_N/images/rgb (T,H,W,3) "
             "for ACT RGBD training. Shares the camera with --save_video.",
    )
    parser.add_argument(
        "--only_success", action="store_true", default=False,
        help="Discard demos where not all picked objects ended up in the basket.",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=0,
        help="Maximum scene attempts before aborting. Use 0 for no limit.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Fixed environment seed. In outer collection mode with multiple demos, "
            "demo i uses seed + i * --seed_stride unless --seeds is provided."
        ),
    )
    parser.add_argument(
        "--seed_stride", type=int, default=1,
        help="Stride used with --seed when collecting multiple demos.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "Explicit seed candidates, one consumed per collection attempt. "
            "This is useful for weight sweeps because every weight can replay "
            "the same candidate scenes while skipping rejected starts."
        ),
    )
    parser.add_argument(
        "--position_priority_orientation_weight",
        "--orientation_weight",
        type=float,
        default=None,
        help=(
            "Override Task E object-1 weighted-pose IK orientation weight for "
            "PRE_GRASP/REACH/CLOSE/LIFT. Defaults to task_e/config.py."
        ),
    )
