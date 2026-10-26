#!/usr/bin/env bash
# Start, or re-attach to, the physcardems container.
# Run from the repository root; it is mounted at /home/shared inside the container.
# To use a differently named container: NAME=mycontainer bash docker/start_docker.sh
NAME=${NAME:-physcardems}
IMAGE=ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

if docker container inspect "$NAME" >/dev/null 2>&1; then
  docker start -ai "$NAME"
else
  docker run -it --name "$NAME" --shm-size=4g \
    -v "$PWD":/home/shared -w /home/shared "$IMAGE"
fi