#!/bin/bash
#
# Local effective dimension vs out-degree for VIBE datasets (src/local_dim_vibe.py), on the robust-prune graphs
# from vibe_pipeline.slurm. effective_dim.ipynb plots the results with the bigann-style ones.
#
# Usage (from the repository root, so slurm_out/ is created next to it):
#   mkdir -p slurm_out
#   sbatch src/local_dim_vibe.slurm <vibe_dataset_name> [...] [-- extra local_dim_vibe.py arguments]
# e.g.
#   sbatch src/local_dim_vibe.slurm landmark-nomic-768-normalized
#   sbatch src/local_dim_vibe.slurm glove-200-cosine yahoo-minilm-384-normalized -- --no-gaussian
#   sbatch src/local_dim_vibe.slurm glove-200-cosine -- --max-points 200000 --data-seed 0 --max-sources 5000
#
# Defaults: a random 10,000 of the sources (VIBE graphs cover every point) and no neighbour-vector files (they would be
# sources x 2048 x d float32: terabytes for all sources). Override after "--", e.g. -- --max-sources 50000.
# Pass the same --max-points / --data-seed (vibe_pipeline's --seed) / --alpha the graph was built with, so the
# points and graph ids match. Datasets without a downloaded .hdf5 or a graph are skipped.
# The job resumes (finished source groups are kept), so after a time limit submit the same command again.
# Block sizes follow the job's memory (--mem) and the GPU's; the VIBE points are held in RAM (n x d float32).
#
# Progress: timestamped lines in slurm_out/slurm_<jobid>.out, one per step and one per minute during each
# pass over the points. Follow with
#   tail -f slurm_out/slurm_<jobid>.out
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=local-dim-vibe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_929_tandon_advanced

REPO="$SCRATCH/multi-vector-search"                  # repository root on the cluster
DOWNLOAD_DIR="$SCRATCH/ANN-Search/datasets/vibe"     # <dataset>.hdf5 files (vibe_pipeline.slurm's DOWNLOAD_DIR)
GRAPH_DIR="$REPO/results/vibe_graphs"                # vibe_pipeline.py's --graph-dir (its default)
LOCAL_PRUNE=1                                        # 1: also --local-prune (out-neighbours in the kNN, robust prune
                                                     #    over N_k(v); re-runs the k-NN pass of results made without it)

# Dataset names up to "--", local_dim_vibe.py arguments after it.
NAMES=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    NAMES+=("$1")
    shift
done
[ "$1" = "--" ] && shift
EXTRA_ARGS="$*"
if [ ${#NAMES[@]} -eq 0 ]; then
    echo "ERROR: give at least one VIBE dataset name."
    echo "Usage: sbatch src/local_dim_vibe.slurm <vibe_dataset_name> [...] [-- extra local_dim_vibe.py arguments]"
    exit 1
fi

DATASET_ARGS=""
for name in "${NAMES[@]}"; do
    DATASET_ARGS="$DATASET_ARGS --dataset='$name'"
done

[ "$LOCAL_PRUNE" = "1" ] && PRUNE_ARGS="--local-prune" || PRUNE_ARGS=""

echo "Starting local-dim VIBE job for: ${NAMES[*]}  (data: $DOWNLOAD_DIR; graphs: $GRAPH_DIR; local prune: $LOCAL_PRUNE; extra args: ${EXTRA_ARGS:-none})"

# --nv exposes the GPU inside the container. PYTHONUNBUFFERED so progress lines reach the log as they happen.
singularity exec --nv --fakeroot --overlay $SCRATCH/envs/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate big_ann
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
nvidia-smi || true

python $REPO/src/local_dim_vibe.py --download-dir='$DOWNLOAD_DIR' --graph-dir='$GRAPH_DIR' $DATASET_ARGS --skip-missing --progress-every 60 $PRUNE_ARGS --max-sources 10000 --no-save-vectors $EXTRA_ARGS
"
