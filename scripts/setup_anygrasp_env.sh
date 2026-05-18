#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ANYGRASP_ENV_NAME:-anygrasp}"
PYTHON_VERSION="${ANYGRASP_PYTHON_VERSION:-3.12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
MAX_JOBS="${MAX_JOBS:-4}"
TORCH_VERSION="${ANYGRASP_TORCH_VERSION:-2.10.0+cu128}"
TORCHVISION_VERSION="${ANYGRASP_TORCHVISION_VERSION:-0.25.0+cu128}"
PYTORCH_INDEX_URL="${ANYGRASP_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANYGRASP_ROOT="${ANYGRASP_ROOT:-${REPO_ROOT}/third_party/anygrasp_sdk}"
ME_ROOT="${ANYGRASP_ROOT}/dependencies/MinkowskiEngine"
OPENSSL11_PREFIX="${ANYGRASP_OPENSSL11_PREFIX:-${REPO_ROOT}/third_party/openssl11}"
CONDA_CHANNEL_FLAGS=(--override-channels -c conda-forge)

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Reusing conda env: ${ENV_NAME}"
else
  echo "[INFO] Creating conda env: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" "${CONDA_CHANNEL_FLAGS[@]}" "python=${PYTHON_VERSION}" \
    pip git cmake ninja setuptools wheel packaging "numpy<2" scipy openblas libopenblas \
    "gcc_linux-64=11.*" "gxx_linux-64=11.*"
fi

echo "[INFO] Ensuring build dependencies are installed from conda-forge"
conda install -y -n "${ENV_NAME}" "${CONDA_CHANNEL_FLAGS[@]}" \
  pip git cmake ninja setuptools wheel packaging "numpy<2" scipy openblas libopenblas \
  "gcc_linux-64=11.*" "gxx_linux-64=11.*"

echo "[INFO] Applying env-local CUDA 12.8 GCC header workaround if needed"
conda run -n "${ENV_NAME}" --no-capture-output bash -lc '
set -euo pipefail
header="${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu/11.4.0/include/c++/bits/shared_ptr_base.h"
if [[ -f "${header}" ]] && grep -q "auto __raw = __to_address(__r.get());" "${header}"; then
  cp -n "${header}" "${header}.anygrasp.bak"
  sed -i "s/auto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/" "${header}"
fi
'

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "[ERROR] nvcc was not found at ${CUDA_HOME}/bin/nvcc"
  echo "        Set CUDA_HOME to the CUDA toolkit path before running this script."
  exit 1
fi

if [[ ! -f "${OPENSSL11_PREFIX}/lib/libcrypto.so.1.1" ]]; then
  echo "[INFO] Creating local OpenSSL 1.1 prefix for AnyGrasp SDK binaries"
  conda create -y -p "${OPENSSL11_PREFIX}" "${CONDA_CHANNEL_FLAGS[@]}" "openssl=1.1.*"
fi

echo "[INFO] Installing PyTorch CUDA 12.8 wheels"
conda run -n "${ENV_NAME}" --no-capture-output python -m pip install --upgrade pip setuptools wheel packaging
conda run -n "${ENV_NAME}" --no-capture-output python -m pip install \
  --progress-bar on --timeout 60 --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

echo "[INFO] Cloning AnyGrasp SDK into ${ANYGRASP_ROOT}"
if [[ ! -d "${ANYGRASP_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${ANYGRASP_ROOT}")"
  git clone https://github.com/graspnet/anygrasp_sdk.git "${ANYGRASP_ROOT}"
else
  git -C "${ANYGRASP_ROOT}" pull --ff-only
fi

echo "[INFO] Cloning modified MinkowskiEngine into ${ME_ROOT}"
mkdir -p "${ANYGRASP_ROOT}/dependencies"
if [[ ! -d "${ME_ROOT}/.git" ]]; then
  git clone https://github.com/chenxi-wang/MinkowskiEngine.git "${ME_ROOT}"
else
  git -C "${ME_ROOT}" pull --ff-only
fi

echo "[INFO] Checking out MinkowskiEngine CUDA 12 branch"
git -C "${ME_ROOT}" checkout cuda-12-1

echo "[INFO] Building MinkowskiEngine with CUDA_HOME=${CUDA_HOME}"
if conda run -n "${ENV_NAME}" python -c "import MinkowskiEngine" >/dev/null 2>&1; then
  echo "[INFO] MinkowskiEngine already imports; skipping rebuild"
else
  conda run -n "${ENV_NAME}" --no-capture-output bash -lc "cd '${ME_ROOT}' && export CUDA_HOME='${CUDA_HOME}' && export MAX_JOBS='${MAX_JOBS}' && export CC=\"\${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc\" && export CXX=\"\${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++\" && export CUDAHOSTCXX=\"\${CXX}\" && export CPATH=\"\${CONDA_PREFIX}/include:\${CPATH:-}\" && export C_INCLUDE_PATH=\"\${CONDA_PREFIX}/include:\${C_INCLUDE_PATH:-}\" && export CPLUS_INCLUDE_PATH=\"\${CONDA_PREFIX}/include:\${CPLUS_INCLUDE_PATH:-}\" && export LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LIBRARY_PATH:-}\" && rm -rf build && python setup.py install --blas_include_dirs='\${CONDA_PREFIX}/include' --blas_library_dirs='\${CONDA_PREFIX}/lib' --blas=openblas"
fi

echo "[INFO] Installing AnyGrasp Python requirements"
conda run -n "${ENV_NAME}" --no-capture-output python -m pip install \
  "numpy<2" Pillow scipy tqdm open3d "transforms3d>=0.4.2" \
  "opencv-python==4.10.0.84" "tifffile<2025" typeguard \
  trimesh scikit-image pywavefront cvxopt dill h5py grasp_nms \
  autolab_core autolab-perception
conda run -n "${ENV_NAME}" --no-capture-output python -m pip install --no-deps graspnetAPI

echo "[INFO] Building pointnet2"
if conda run -n "${ENV_NAME}" python -c "import pointnet2._ext" >/dev/null 2>&1; then
  echo "[INFO] pointnet2 already imports; skipping rebuild"
else
  conda run -n "${ENV_NAME}" --no-capture-output bash -lc "cd '${ANYGRASP_ROOT}/pointnet2' && export CUDA_HOME='${CUDA_HOME}' && export MAX_JOBS='${MAX_JOBS}' && export CC=\"\${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc\" && export CXX=\"\${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++\" && export CUDAHOSTCXX=\"\${CXX}\" && python setup.py install"
fi

PY_ABI="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(f"cpython-{sys.version_info.major}{sys.version_info.minor}")' | tr -d '[:space:]')"
echo "[INFO] Installing AnyGrasp SDK shared libraries for ${PY_ABI}"
cp "${ANYGRASP_ROOT}/grasp_detection/gsnet_versions/gsnet.${PY_ABI}-x86_64-linux-gnu.so" \
  "${ANYGRASP_ROOT}/grasp_detection/gsnet.so"
cp "${ANYGRASP_ROOT}/license_registration/lib_cxx_versions/lib_cxx.${PY_ABI}-x86_64-linux-gnu.so" \
  "${ANYGRASP_ROOT}/grasp_detection/lib_cxx.so"
cp "${ANYGRASP_ROOT}/grasp_tracking/tracker_versions/tracker.${PY_ABI}-x86_64-linux-gnu.so" \
  "${ANYGRASP_ROOT}/grasp_tracking/tracker.so"
cp "${ANYGRASP_ROOT}/license_registration/lib_cxx_versions/lib_cxx.${PY_ABI}-x86_64-linux-gnu.so" \
  "${ANYGRASP_ROOT}/grasp_tracking/lib_cxx.so"

echo "[INFO] Verifying core imports"
conda run -n "${ENV_NAME}" python -c "import torch, numpy, cv2, MinkowskiEngine as ME, open3d, transforms3d, graspnetAPI, pointnet2, pointnet2._ext; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('numpy', numpy.__version__); print('cv2', cv2.__version__); print('MinkowskiEngine', ME.__version__); print('open3d', open3d.__version__); print('transforms3d', transforms3d.__version__); print('graspnetAPI import ok'); print('pointnet2 import ok')"

cat <<EOF

[DONE] AnyGrasp env installed.

Next manual steps:
1. Activate the env:
   conda activate ${ENV_NAME}
2. Register the AnyGrasp license:
   cd ${ANYGRASP_ROOT}/license_registration
   LD_LIBRARY_PATH=${OPENSSL11_PREFIX}/lib:\$LD_LIBRARY_PATH ./license_checker -f
3. Submit the feature id to the AnyGrasp license form.
4. Unzip the returned license folder into:
   ${ANYGRASP_ROOT}/grasp_detection/license
   ${ANYGRASP_ROOT}/grasp_tracking/license
5. The Python ${PYTHON_VERSION} SDK .so files were already copied into grasp_detection and grasp_tracking.

EOF
