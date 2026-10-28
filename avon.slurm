#!/bin/bash -l
# Experiments of the note and of its supplementary material on Avon: an array of one task per configuration in
# experiment.py of the group of experiments of config named by the argument, each on 16 cores of a compute node shared
# with other jobs, with one process per core, so that the tasks run at the same time on the free cores of several
# nodes. The memory is requested per core, since Avon gives the whole memory of the node to a job which does not
# request any, and the past runs used about 0.5 GB per process. The tests run locally, not on Avon.
# Submit from the root of the clone, after avon_setup.sh, with one task per configuration (six numbers of
# particles per experiment):
#   sbatch --array=0-23 avon.slurm EXPERIMENTS
#   sbatch --array=0-11 avon.slurm LOCALLY_OPTIMAL_EXPERIMENTS
#   sbatch --array=0-23 avon.slurm STRATIFIED_EXPERIMENTS
#   sbatch --array=0-11 avon.slurm STRATIFIED_LOCALLY_OPTIMAL_EXPERIMENTS
#SBATCH --job-name=metropolized-bs
#SBATCH --account=stats
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source avon_env.sh
source "$VENV/bin/activate"
# index in experiment.CONFIGURATIONS of the configuration of this task, which fails unless the array has one task per
# configuration of the group, so that every configuration runs and none of the results comes from an earlier run
INDEX=$(python3 -c "
import os, sys, config, experiment
indices = experiment.configuration_indices(getattr(config, sys.argv[1]))
assert len(indices) == int(os.environ['SLURM_ARRAY_TASK_COUNT']), 'the array must have one task per configuration'
print(indices[int(os.environ['SLURM_ARRAY_TASK_ID'])])
" "$1")
python3 experiment.py "$SLURM_CPUS_PER_TASK" "$INDEX"
