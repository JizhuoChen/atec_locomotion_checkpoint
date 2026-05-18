# AnyGrasp Environment Setup

This repo keeps AnyGrasp isolated from `isaaclab` and `sam3_full`.

## Target Environment

- Conda env: `anygrasp`
- Python: `3.12`
- CUDA runtime: `12.8`
- PyTorch: `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`
- AnyGrasp SDK checkout: `third_party/anygrasp_sdk`
- MinkowskiEngine checkout: `third_party/anygrasp_sdk/dependencies/MinkowskiEngine`
- OpenSSL 1.1 helper prefix for vendor license binaries: `third_party/openssl11`

The current `sam3_full` env reports:

```bash
python 3.12.13
torch 2.10.0+cu128
torch cuda 12.8
cuda available True
```

## One-Command Install

Run from the repo root:

```bash
bash scripts/setup_anygrasp_env.sh
```

Optional overrides:

```bash
ANYGRASP_ENV_NAME=anygrasp \
ANYGRASP_ROOT=/home/steven/Projects/Projects/anygrasp_sdk \
CUDA_HOME=/usr/local/cuda-12 \
MAX_JOBS=4 \
bash scripts/setup_anygrasp_env.sh
```

The script creates a separate conda env, installs PyTorch CUDA 12.8, clones the official AnyGrasp SDK, builds the modified MinkowskiEngine branch for CUDA 12, installs AnyGrasp requirements, and builds `pointnet2`.
It also copies the Python 3.12 SDK libraries to `grasp_detection/gsnet.so`, `grasp_detection/lib_cxx.so`, `grasp_tracking/tracker.so`, and `grasp_tracking/lib_cxx.so`.

## Manual Install Equivalent

```bash
conda create -n anygrasp --override-channels -c conda-forge \
  python=3.12 pip git cmake ninja setuptools wheel packaging \
  "numpy<2" scipy openblas libopenblas "gcc_linux-64=11.*" "gxx_linux-64=11.*"
conda activate anygrasp

pip install --upgrade pip setuptools wheel packaging
pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"

mkdir -p third_party
git clone https://github.com/graspnet/anygrasp_sdk.git third_party/anygrasp_sdk
cd third_party/anygrasp_sdk

mkdir -p dependencies
git clone https://github.com/chenxi-wang/MinkowskiEngine.git dependencies/MinkowskiEngine
cd dependencies/MinkowskiEngine
git checkout cuda-12-1
export CUDA_HOME=/usr/local/cuda-12
export MAX_JOBS=4
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas_library_dirs=${CONDA_PREFIX}/lib --blas=openblas

cd ../..
pip install "numpy<2" Pillow scipy tqdm open3d "transforms3d>=0.4.2" \
  "opencv-python==4.10.0.84" "tifffile<2025" typeguard \
  trimesh scikit-image pywavefront cvxopt dill h5py grasp_nms \
  autolab_core autolab-perception
pip install --no-deps graspnetAPI

cd pointnet2
python setup.py install
```

## License Registration

AnyGrasp SDK needs a machine-bound license.

```bash
conda activate anygrasp
cd third_party/anygrasp_sdk/license_registration
LD_LIBRARY_PATH=$PWD/../../openssl11/lib:$LD_LIBRARY_PATH ./license_checker -f
```

From the repository root you can use the wrapper instead:

```bash
scripts/anygrasp_license_checker.sh -f
```

On this machine the feature id printed by the checker is:

```text
2028705673378424846
```

Submit the feature id to the AnyGrasp license form linked in the official README. If the printed id ends with `%`, remove the `%` before submitting.

When the license zip is returned, unzip it so the folder is available as:

```text
third_party/anygrasp_sdk/grasp_detection/license
third_party/anygrasp_sdk/grasp_tracking/license
```

Or install the returned artifacts into the expected SDK layout with:

```bash
python scripts/install_anygrasp_artifacts.py \
  --license-zip /path/to/returned_license.zip \
  --checkpoint /path/to/checkpoint_detection.tar
```

If the license is already extracted:

```bash
python scripts/install_anygrasp_artifacts.py \
  --license-dir /path/to/license \
  --checkpoint /path/to/checkpoint_detection.tar
```

You can check it with:

```bash
cd third_party/anygrasp_sdk/license_registration
LD_LIBRARY_PATH=$PWD/../../openssl11/lib:$LD_LIBRARY_PATH \
  ./license_checker -c ../grasp_detection/license/licenseCfg.json
```

Until the license folder is installed, importing `gsnet.so` may print `license failed to pass`; that means the binary is found but the machine-bound license is still missing.

## Python-Version SDK Libraries

The SDK uses binary `.so` libraries. If `grasp_detection` asks for `gsnet.so` or `lib_cxx.so`, copy the files matching the env Python version into the demo folder. For Python 3.12, choose the `cpython-312` files if present in the SDK checkout.

Example pattern:

```bash
cd third_party/anygrasp_sdk/grasp_detection
cp gsnet_versions/gsnet.cpython-312-*.so gsnet.so
cp ../license_registration/lib_cxx_versions/lib_cxx.cpython-312-*.so lib_cxx.so
```

The setup script already performs these copies for detection and tracking.

## CUDA 12.8 Compile Note

The official AnyGrasp README notes a CUDA 12.8 workaround for a `std::__to_address` compile error in MinkowskiEngine. The setup script applies that workaround only to the env-local GCC 11 header:

```text
$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/11.4.0/include/c++/bits/shared_ptr_base.h
```

Do not patch `/usr/include` for this setup. Prefer first confirming:

```bash
which nvcc
nvcc --version
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

If MinkowskiEngine fails, keep the exact compiler error and fix only that failure.
