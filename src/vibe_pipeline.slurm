#!/bin/bash
#
# VIBE dataset vs a same-spectrum Gaussian: robust-prune graphs + statistics (src/vibe_pipeline.py).
#
# Usage (from the repository root, so slurm_out/ is created next to it):
#   mkdir -p slurm_out
#   sbatch src/vibe_pipeline.slurm <dataset_name> [extra vibe_pipeline.py arguments]
# e.g.
#   sbatch src/vibe_pipeline.slurm landmark-nomic-768-normalized
#   sbatch src/vibe_pipeline.slurm glove-200-cosine --max-points 200000 --batch-size 1024
#
# Every step resumes, so after a time limit just submit the same command again.
# If compute nodes have no internet, download first on a login node (same container + env):
#   python src/vibe_pipeline.py --dataset <dataset_name> --download-dir $SCRATCH/vibe --download-only
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=vibe-prune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_929_tandon_advanced

if [ -z "$1" ]; then
    echo "ERROR: Please provide the dataset name as a command-line argument."
    echo "Usage: sbatch src/vibe_pipeline.slurm <dataset_name> [extra vibe_pipeline.py arguments]"
    exit 1
fi

DATASET_NAME="$1"
shift
EXTRA_ARGS="$*"

REPO="$SCRATCH/multi-vector-search"        # repository root on the cluster
DOWNLOAD_DIR="$SCRATCH/ANN-Search/datasets/vibe"               # where the .hdf5 files are kept

echo "Starting job for dataset: $DATASET_NAME  (extra args: ${EXTRA_ARGS:-none})"

# --nv exposes the GPU inside the container. Threads for the NumPy covariance / CPU fallback.
singularity exec --nv --fakeroot --overlay $SCRATCH/envs/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate big_ann
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi || true

python $REPO/src/vibe_pipeline.py --dataset='$DATASET_NAME' --download-dir='$DOWNLOAD_DIR' $EXTRA_ARGS
"
