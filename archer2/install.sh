#!/usr/bin/env bash
# One-off ARCHER2 setup. Run on a LOGIN node (compute nodes have no internet),
# from the repo root:   bash archer2/install.sh
# Pulls the dolfinx container, then creates a virtual environment on /work that also
# sees the container's packages, and installs the cardiac libraries and physcardems into it.
# Everything lands on /work, which compute nodes also mount.
(
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$(dirname "$PCE_IMAGE")" "$PCE_CACHE"
export SINGULARITY_CACHEDIR="$PCE_CACHE/singularity"

if [ ! -f "$PCE_IMAGE" ]; then
  singularity pull "$PCE_IMAGE" docker://ghcr.io/fenics/dolfinx/dolfinx:v0.10.0
fi

singularity exec -B "$PCE_WORK" "$PCE_IMAGE" env \
  HOME="$PCE_WORK" XDG_CACHE_HOME="$PCE_CACHE" PIP_CACHE_DIR="$PCE_CACHE/pip" \
  PCE_VENV="$PCE_VENV" PCE_REPO="$PCE_REPO" \
  bash -c '
set -euo pipefail
purelib() { "$1" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])"; }
if [ ! -x "$PCE_VENV/bin/python3" ]; then
  base_site=$(purelib python3)
  python3 -m venv "$PCE_VENV"
  echo "$base_site" > "$(purelib "$PCE_VENV/bin/python3")/_container_env.pth"
  echo "Created $PCE_VENV on top of $base_site"
fi
export PATH="$PCE_VENV/bin:$PATH" VIRTUAL_ENV="$PCE_VENV"
bash "$PCE_REPO/docker/setup.sh"
python3 -m pip install scipy
'
echo "ARCHER2 install done: image $PCE_IMAGE, environment $PCE_VENV"
)