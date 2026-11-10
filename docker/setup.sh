#!/usr/bin/env bash
# Run once inside a new container (or from archer2/install.sh). Works from any folder,
# and whether run (bash setup.sh) or sourced (. setup.sh).
# Installs simcardemsx (the computational back end) from a pinned commit, the pulse and
# beat versions it requires, and physcardems itself, editable.
# The body runs in a subshell, so `set -e` and the `cd` never leak into your shell.
(
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "Installing into $REPO"

SIMCARDEMSX_COMMIT="${SIMCARDEMSX_COMMIT:-e1d6a1c46cff7320238a1b2f60101ceb7ce346fa}"

# System libraries needed by gmsh (used by cardiac-geometries). Only possible as root,
# so skipped inside Singularity on ARCHER2.
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  apt-get update && apt-get install -y --no-install-recommends \
      libxrender1 libxcursor1 libxft2 libxinerama1 libglu1-mesa
fi

# scifem must be built without build isolation, so install its build backend first
python3 -m pip install "scikit-build-core[pyproject]>=0.11.0" "nanobind>=1.3.2" \
    "setuptools>=77.0.3" "packaging>=24.2" pkgconfig
python3 -m pip install scifem==0.22.1 --no-build-isolation

# simcardemsx at the pinned commit; it pulls in the pulse, beat, crossbridge and
# gotranx versions it needs
python3 -m pip install "simcardemsx @ git+https://github.com/ComputationalPhysiology/simcardemsx.git@${SIMCARDEMSX_COMMIT}"

# pulse's demo extras (circulation, cardiac-geometriesx), at the version already installed
python3 -m pip install "fenicsx-pulse[demo]"

# physcardems itself, editable
python3 -m pip install -e "$REPO"

# Record exactly what was installed
mkdir -p "$REPO/environment"
python3 -m pip freeze --exclude-editable > "$REPO/environment/pip-freeze-simcardemsx.txt"

python3 -c "import pulse, beat, simcardemsx, gotranx, cardiac_geometries, circulation, physcardems; \
print('pulse', pulse.__version__, '| beat', beat.__version__, '| gotranx', gotranx.__version__, \
'| simcardemsx', simcardemsx.__version__)"
)