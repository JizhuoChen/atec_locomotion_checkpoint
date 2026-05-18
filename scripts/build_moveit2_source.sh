#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOVEIT_DIR="${REPO_ROOT}/third_party/moveit2"
WS_DIR="${REPO_ROOT}/third_party/moveit2_ws"

if [[ ! -d "${MOVEIT_DIR}" ]]; then
  echo "[ERROR] Missing ${MOVEIT_DIR}. Run scripts/setup_moveit2_source.sh first." >&2
  exit 1
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "[ERROR] ROS Jazzy setup file not found: /opt/ros/jazzy/setup.bash" >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash

if ! rosdep db >/dev/null 2>&1; then
  cat >&2 <<'EOF'
[ERROR] rosdep is not initialized.
Run once with sudo privileges:
  sudo rosdep init
  rosdep update
EOF
  exit 1
fi

rosdep install --from-paths "${MOVEIT_DIR}" --ignore-src -r -y --rosdistro jazzy

mkdir -p "${WS_DIR}"
colcon build \
  --base-paths "${MOVEIT_DIR}" \
  --build-base "${WS_DIR}/build" \
  --install-base "${WS_DIR}/install" \
  --log-base "${WS_DIR}/log" \
  --packages-up-to moveit_py

echo "[INFO] MoveIt2 workspace installed at ${WS_DIR}/install"
echo "[INFO] Source it with: source ${WS_DIR}/install/setup.bash"
