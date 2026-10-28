#!/bin/bash -l
# One-off setup on Avon, from the root of the clone: creates the Python virtual environment with the
# pinned requirements with uv, and the directory of the Slurm logs. All the computations run through Slurm
# (avon.slurm), never on the login node.
set -euo pipefail
source avon_env.sh
uv venv --python "$(command -v python3)" "$VENV"
uv pip install --python "$VENV/bin/python" -r requirements.txt
mkdir -p logs
