# Environment of the experiments on the Avon cluster (SCRTP, University of Warwick): the modules which
# provide Python, and the Python virtual environment created by avon_setup.sh. Sourced by avon_setup.sh
# and avon.slurm.
module purge
module load GCCcore/13.3.0 Python/3.12.3
VENV="$HOME/venvs/metropolized-bs"
