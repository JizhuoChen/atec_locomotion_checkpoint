#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOVEIT_DIR="${REPO_ROOT}/third_party/moveit2"

if [[ -d "${MOVEIT_DIR}/.git" ]]; then
  echo "[INFO] MoveIt2 source already exists: ${MOVEIT_DIR}"
  git -C "${MOVEIT_DIR}" rev-parse --short HEAD
  exit 0
fi

mkdir -p "${REPO_ROOT}/third_party"
git clone --depth 1 https://github.com/moveit/moveit2.git "${MOVEIT_DIR}"
git -C "${MOVEIT_DIR}" rev-parse --short HEAD
