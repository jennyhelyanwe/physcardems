#!/usr/bin/env bash
# Start (or reattach to) the physcardems container, with simcardemsx as the back end.
# Run from the repo root. First time only, inside the container: bash docker/setup.sh
NAME=${NAME:-physcardems-simcardemsx}
IMAGE=ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

if docker container inspect "$NAME" >/dev/null 2>&1; then
  docker start -ai "$NAME"
else
  docker run -it --name "$NAME" --shm-size=4g \
    -v "$PWD":/home/shared -w /home/shared "$IMAGE"
fi