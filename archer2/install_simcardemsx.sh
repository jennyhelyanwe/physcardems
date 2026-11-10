#!/usr/bin/env bash
# One-off: a separate venv with simcardemsx main and the pulse/beat versions it needs,
# on the same dolfinx container as physcardems. Leaves $PCE_VENV untouched.
#   bash archer2/install_simcardemsx.sh
(
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SIMCARDEMSX_VENV="${SIMCARDEMSX_VENV:-$PCE_WORK/simcardemsx_venv}"
SIMCARDEMSX_SRC="${SIMCARDEMSX_SRC:-$PCE_WORK/simcardemsx}"

[ -d "$SIMCARDEMSX_SRC" ] || git clone https://github.com/ComputationalPhysiology/simcardemsx.git "$SIMCARDEMSX_SRC"
git -C "$SIMCARDEMSX_SRC" pull --ff-only
echo "simcardemsx at $(git -C "$SIMCARDEMSX_SRC" rev-parse --short HEAD)"

singularity exec -B "$PCE_WORK" "$PCE_IMAGE" env \
  HOME="$PCE_WORK" XDG_CACHE_HOME="$PCE_CACHE" PIP_CACHE_DIR="$PCE_CACHE/pip" \
  SIMCARDEMSX_VENV="$SIMCARDEMSX_VENV" SIMCARDEMSX_SRC="$SIMCARDEMSX_SRC" \
  bash -c '
set -euo pipefail
purelib() { "$1" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])"; }
if [ ! -x "$SIMCARDEMSX_VENV/bin/python3" ]; then
  base_site=$(purelib python3)
  python3 -m venv "$SIMCARDEMSX_VENV"
  echo "$base_site" > "$(purelib "$SIMCARDEMSX_VENV/bin/python3")/_container_env.pth"
fi
export PATH="$SIMCARDEMSX_VENV/bin:$PATH" VIRTUAL_ENV="$SIMCARDEMSX_VENV"
python3 -m pip install "scikit-build-core[pyproject]>=0.11.0" "nanobind>=1.3.2" \
    "setuptools>=77.0.3" "packaging>=24.2" pkgconfig
python3 -m pip install scifem==0.22.1 --no-build-isolation
python3 -m pip install -e "$SIMCARDEMSX_SRC" pandas matplotlib
python3 -c "import simcardemsx, pulse, beat, gotranx; \
print(\"pulse\", pulse.__version__, \"| beat\", beat.__version__, \"| gotranx\", gotranx.__version__)"
'
)