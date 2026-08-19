"""Privileged teacher/student extension of the canonical robust B2-Piper task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.robust_piper_env_cfg import (
    UnitreeB2PiperRobustHeadingRoughEnvCfg,
)
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import ObservationsCfg


LEG_FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


@configclass
class TeacherPrivilegedObservationsCfg(ObsGroup):
    """Clean simulator-only state appended after the student's 45 observations."""

    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        clip=(-10.0, 10.0),
        scale=2.0,
    )
    arm_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "arm_joint1",
                    "arm_joint2",
                    "arm_joint3",
                    "arm_joint4",
                    "arm_joint5",
                    "arm_joint6",
                    "arm_joint7",
                    "arm_joint8",
                ],
                preserve_order=True,
            )
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    arm_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "arm_joint1",
                    "arm_joint2",
                    "arm_joint3",
                    "arm_joint4",
                    "arm_joint5",
                    "arm_joint6",
                    "arm_joint7",
                    "arm_joint8",
                ],
                preserve_order=True,
            )
        },
        clip=(-100.0, 100.0),
        scale=0.05,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ContactForceObservationsCfg(ObsGroup):
    """Four foot-force vectors shared by the privileged teacher and critic."""

    foot_contact_forces = ObsTerm(
        func=mdp.contact_forces_b,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=LEG_FOOT_NAMES,
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
        },
        # A 100 N component maps to 1.0.  Clipping bounds rare impact spikes
        # without discarding the direction of the base-frame force vector.
        clip=(-500.0, 500.0),
        scale=0.01,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TeacherHeightScanObservationsCfg(ObsGroup):
    """Ground-truth terrain-height map for the privileged teacher."""

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TeacherStudentObservationsCfg(ObservationsCfg):
    """Canonical observations plus explicit teacher-only observation groups."""

    teacher_privileged: TeacherPrivilegedObservationsCfg = TeacherPrivilegedObservationsCfg()
    contact_forces: ContactForceObservationsCfg = ContactForceObservationsCfg()
    teacher_height_scan: TeacherHeightScanObservationsCfg = TeacherHeightScanObservationsCfg()


@configclass
class UnitreeB2PiperRobustHeadingRoughTeacherStudentEnvCfg(UnitreeB2PiperRobustHeadingRoughEnvCfg):
    """Robust B2-Piper task exposing all teacher/student observation groups.

    The original ``policy`` group remains the unchanged 45-input deployable
    interface.  RSL-RL runner configuration decides whether the extra groups
    feed the 76-input legacy teacher, the 263-input height-scan teacher, or are
    ignored by a deployed student.  Both teacher variants use a 263-input
    privileged critic.
    """

    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()
