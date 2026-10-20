#!/usr/bin/env bash
# Run once inside a new container. Works from any folder, including docker/,
# and whether run (bash setup.sh) or sourced (. setup.sh).
# The body runs in a subshell, so `set -e` and the `cd` never leak into your shell.
(
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "Installing into $REPO"

# System libraries needed by gmsh (used by cardiac-geometries)
apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxcursor1 libxft2 libxinerama1 libglu1-mesa

# scifem must be built without build isolation, so install its build backend first
# (listed explicitly; scifem removed build-requirements.txt from main in Sept 2026)
python3 -m pip install "scikit-build-core[pyproject]>=0.11.0" "nanobind>=1.3.2" \
    "setuptools>=77.0.3" "packaging>=24.2" pkgconfig
python3 -m pip install scifem==0.22.1 --no-build-isolation

# Cardiac libraries, pinned to the tested commits (see environment/pip-freeze.txt)
python3 -m pip install "fenicsx-pulse[demo] @ git+https://github.com/finsberg/fenicsx-pulse.git@5dfeabca25bd7f28479741a1b2e965a01c83d01b"
python3 -m pip install "fenicsx-beat @ git+https://github.com/finsberg/fenicsx-beat.git@0c455421b4f5b26b5dc4d1488ccdb04b5c831878"
python3 -m pip install "simcardemsx @ git+https://github.com/ComputationalPhysiology/simcardemsx.git@0bccb0d02567add31f57a540972603cb95eaa080"

# physcardems itself, editable
python3 -m pip install -e "$REPO"

python3 -c "import pulse, beat, simcardemsx, gotranx, cardiac_geometries, circulation, physcardems; \
print('pulse', pulse.__version__, '| beat', beat.__version__, '| gotranx', gotranx.__version__)"
)