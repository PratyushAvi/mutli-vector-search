#!/bin/bash
#
# Local effective dimension vs out-degree (src/local_dim_pipeline.py). effective_dim.ipynb plots the results.
#
# Usage (from the repository root, so slurm_out/ is created next to it):
#   mkdir -p slurm_out
#   sbatch src/local_dim_pipeline.slurm [<dataset_name> ...] [-- extra local_dim_pipeline.py arguments]
# e.g.
#   sbatch src/local_dim_pipeline.slurm                                  # all of DEFAULT_DATASETS
#   sbatch src/local_dim_pipeline.slurm bigann spacev1b
#   sbatch src/local_dim_pipeline.slurm bigann -- --rows bigann=10000000 --max-sources 5000
#
# Names are folders in DATASETS_DIR (each holding vectors.npy); graphs are read from GRAPH_DIR
# (adj-list-<name>.txt and <name>-computed.txt). Datasets whose graph files are missing are skipped.
# The job resumes (finished source groups are kept), so after a time limit submit the same command again.
# Block sizes follow the job's memory (--mem, read through SLURM_MEM_PER_NODE / the cgroup) and the GPU's.
#
# Progress: timestamped lines in slurm_out/slurm_<jobid>.out, one per step and one per minute during each
# pass over vectors.npy (blocks done / total, elapsed, time left, rate). Follow with
#   tail -f slurm_out/slurm_<jobid>.out
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=local-dim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_929_tandon_advanced

REPO="$SCRATCH/multi-vector-search"                  # repository root on the cluster
DATASETS_DIR="$SCRATCH/ANN-Search/datasets"          # dataset folders (vectors.npy)
GRAPH_DIR="$SCRATCH/ANN-Search/navigable_graph_results/new_results"               # TODO: where adj-list-<name>.txt and <name>-computed.txt are
DEFAULT_DATASETS=(bigann spacev1b "facebook_sim_searchnet++" yandex_deep)
LOCAL_PRUNE=1                                        # 1: also --local-prune (out-neighbours in the kNN, robust prune
                                                     #    over N_k(v); re-runs the k-NN pass of results made without it)

# Dataset names up to "--", local_dim_pipeline.py arguments after it.
NAMES=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    NAMES+=("$1")
    shift
done
[ "$1" = "--" ] && shift
EXTRA_ARGS="$*"
[ ${#NAMES[@]} -eq 0 ] && NAMES=("${DEFAULT_DATASETS[@]}")

DATASET_ARGS=""
for name in "${NAMES[@]}"; do
    if [ ! -f "$DATASETS_DIR/$name/vectors.npy" ]; then
        echo "ERROR: $DATASETS_DIR/$name/vectors.npy not found"
        exit 1
    fi
    DATASET_ARGS="$DATASET_ARGS --dataset='$DATASETS_DIR/$name'"
done

[ "$LOCAL_PRUNE" = "1" ] && PRUNE_ARGS="--local-prune" || PRUNE_ARGS=""

echo "Starting local-dim job for: ${NAMES[*]}  (graphs: $GRAPH_DIR; local prune: $LOCAL_PRUNE; extra args: ${EXTRA_ARGS:-none})"

# --nv exposes the GPU inside the container. PYTHONUNBUFFERED so progress lines reach the log as they happen.
singularity exec --nv --fakeroot --overlay $SCRATCH/envs/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate big_ann
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
nvidia-smi || true

python $REPO/src/local_dim_pipeline.py --graph-dir='$GRAPH_DIR' $DATASET_ARGS --skip-missing --progress-every 60 $PRUNE_ARGS $EXTRA_ARGS
"
