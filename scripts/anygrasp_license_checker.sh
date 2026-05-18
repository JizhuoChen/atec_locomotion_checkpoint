#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER_DIR="${REPO_ROOT}/third_party/anygrasp_sdk/license_registration"
OPENSSL11_LIB="${REPO_ROOT}/third_party/openssl11/lib"

if [[ ! -x "${CHECKER_DIR}/license_checker" ]]; then
  echo "[ERROR] Missing AnyGrasp license_checker at ${CHECKER_DIR}/license_checker" >&2
  exit 2
fi

if [[ ! -f "${OPENSSL11_LIB}/libcrypto.so.1.1" ]]; then
  echo "[ERROR] Missing libcrypto.so.1.1 at ${OPENSSL11_LIB}" >&2
  echo "Run: conda create -y -p ${REPO_ROOT}/third_party/openssl11 --override-channels -c conda-forge 'openssl=1.1.*'" >&2
  exit 2
fi

cd "${CHECKER_DIR}" || exit 2

# The vendor checker is brittle: it can return 1 after printing the feature id,
# and it can segfault if stdout is captured or redirected. For registration,
# call it directly and normalize the exit code for `-f`.
if [[ "$*" == *"-f"* ]]; then
  LD_LIBRARY_PATH="${OPENSSL11_LIB}:${LD_LIBRARY_PATH:-}" ./license_checker "$@"
  status=$?
  if [[ "${status}" -eq 1 ]]; then
    exit 0
  fi
  exit "${status}"
fi

LD_LIBRARY_PATH="${OPENSSL11_LIB}:${LD_LIBRARY_PATH:-}" ./license_checker "$@"
status=$?
exit "${status}"
