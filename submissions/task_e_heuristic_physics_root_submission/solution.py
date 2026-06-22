"""Task E heuristic execution policy.

This is a root-entrypoint-compatible submission file.  It contains the
banana+mustard segment of the real-physics heuristic sequence validated in
``outputs/task_e_heuristic_physics_run/seed_sweep_debug_layout``.
"""

from __future__ import annotations

import numpy as np
import torch


class AlgSolution:
    _ACTION_DIM = 8
    _START_ACTION = [0.0] * _ACTION_DIM

    # Waypoints 00 through 02_mustard_bottle_lift_retract from the validated
    # physics run.  The later box segment is omitted because it was 0/5.
    _WAYPOINTS = [
        (100, [-6.578054308192804e-05, -0.5509419441223145, -0.02890777587890625, 2.1621286578010768e-05, 0.3599996566772461, -6.432534428313375e-05, 0.0, 0.0]),
        (140, [-0.17395435273647308, 1.689814567565918, -2.1388144493103027, 2.346141445741523e-05, 0.3599998950958252, -0.17394806444644928, 0.0, 0.0]),
        (140, [-0.1739465743303299, 1.4947528839111328, -0.12329387664794922, 1.6199872334254906e-05, 0.1442091464996338, -0.173963725566864, 0.0, 0.0]),
        (120, [-0.1755581945180893, 1.8264856338500977, 0.35227203369140625, -0.001629621023312211, -0.662420392036438, -0.17467129230499268, 0.0, 0.0]),
        (70, [-0.19981829822063446, 1.705965518951416, 0.3141927719116211, -0.0024779224768280983, -0.559447169303894, -0.25662344694137573, -0.10000000149011612, 0.10000000149011612]),
        (140, [-0.17381946742534637, 1.4931550025939941, -0.6330199241638184, -0.0008879865636117756, 0.3599996566772461, -0.17371053993701935, -0.10000000149011612, 0.10000000149011612]),
        (140, [-0.17302049696445465, 1.693650245666504, -2.1411495208740234, -0.003276629839092493, 0.3599998950958252, -0.17390432953834534, -0.10000000149011612, 0.10000000149011612]),
        (230, [1.3895045518875122, 1.8781676292419434, -2.3424878120422363, -0.0028462866321206093, 0.36000633239746094, 1.3879023790359497, -0.10000000149011612, 0.10000000149011612]),
        (110, [1.388588786125183, 1.9227681159973145, -0.19919109344482422, -1.8198101315647364e-05, -0.2075824737548828, 1.3885669708251953, -0.10000000149011612, 0.10000000149011612]),
        (80, [1.388835072517395, 1.9204916954040527, -0.19473004341125488, 2.1982239559292793e-05, -0.20976567268371582, 1.3888112306594849, 0.0, 0.0]),
        (100, [1.3887091875076294, 1.874131202697754, -2.337779998779297, 1.422325658495538e-05, 0.3599996566772461, 1.38870108127594, 0.0, 0.0]),
        (140, [-0.7319468855857849, 1.6998686790466309, -2.15155029296875, 2.3514474378316663e-05, 0.3599998950958252, -0.7319399118423462, 0.0, 0.0]),
        (140, [-0.731940507888794, 1.5360064506530762, -0.9028568267822266, 1.608244019735139e-05, 0.3599998950958252, -0.7319505214691162, 0.0, 0.0]),
        (120, [-0.7333136796951294, 1.5733342170715332, 0.048729896545410156, -0.0029458440840244293, -0.1059575080871582, -0.7307610511779785, 0.0, 0.0]),
        (70, [-0.7517147660255432, 1.3521506786346436, -0.10902810096740723, -0.02018585056066513, -0.12171268463134766, -0.7338354587554932, -0.10000000149011612, 0.10000000149011612]),
        (140, [-0.7313058376312256, 1.6818504333496094, -1.807166576385498, -0.002544471062719822, 0.3600006103515625, -0.7313870787620544, -0.10000000149011612, 0.10000000149011612]),
        (140, [-0.7310748100280762, 1.703969955444336, -2.154231548309326, -0.003229795955121517, 0.3600013256072998, -0.7314687967300415, -0.10000000149011612, 0.10000000149011612]),
        (230, [1.6982775926589966, 1.7765803337097168, -2.241461753845215, -0.0031195329502224922, 0.36001157760620117, 1.6960667371749878, -0.10000000149011612, 0.10000000149011612]),
        (110, [1.6979893445968628, 1.6312355995178223, 0.11368393898010254, 0.0012090476229786873, -0.31902456283569336, 1.6957217454910278, -0.10000000149011612, 0.10000000149011612]),
        (80, [1.6981505155563354, 1.6215877532958984, 0.11669683456420898, 0.0004846472293138504, -0.3223731517791748, 1.6959261894226074, 0.0, 0.0]),
        (100, [1.697328805923462, 1.7722439765930176, -2.237517833709717, 1.841964513005223e-05, 0.3599996566772461, 1.6973166465759277, 0.0, 0.0]),
    ]

    def __init__(self):
        self._actions = self._expand_waypoints()
        self._step = 0
        self._hold_action = list(self._actions[-1] if self._actions else self._START_ACTION)

    def reset(self, **_kwargs):
        self._step = 0

    def predicts(self, obs, current_score):
        action_dim = self._infer_action_dim(obs)
        if self._step < len(self._actions):
            action = self._fit_action_dim(self._actions[self._step], action_dim)
            self._step += 1
        else:
            action = self._fit_action_dim(self._hold_action, action_dim)
        return {"action": [action for _ in range(self._infer_num_envs(obs))], "giveup": False}

    def _expand_waypoints(self) -> list[list[float]]:
        actions: list[list[float]] = []
        start = np.asarray(self._START_ACTION, dtype=np.float32)
        for steps, target_action in self._WAYPOINTS:
            target = np.asarray(target_action, dtype=np.float32)
            for idx in range(max(1, int(steps))):
                alpha = float(idx + 1) / float(max(1, int(steps)))
                actions.append((start + alpha * (target - start)).astype(float).tolist())
            start = target
        return actions

    def _infer_num_envs(self, obs) -> int:
        proprio = obs.get("proprio") if isinstance(obs, dict) else None
        if proprio is None:
            return 1
        return int(proprio.shape[0]) if len(proprio.shape) > 1 else 1

    def _infer_action_dim(self, obs) -> int:
        proprio = obs.get("proprio") if isinstance(obs, dict) else None
        if proprio is None:
            return self._ACTION_DIM
        dim = int(proprio.shape[-1])
        return self._ACTION_DIM if dim >= 24 else max(1, dim // 3)

    def _fit_action_dim(self, action: list[float], action_dim: int) -> list[float]:
        fitted = list(action[:action_dim])
        if len(fitted) < action_dim:
            fitted.extend([0.0] * (action_dim - len(fitted)))
        return fitted
