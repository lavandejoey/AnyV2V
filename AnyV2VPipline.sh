#!/bin/bash
#SBATCH --job-name=AnyV2V
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=L40S
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
umask 077
mkdir -p logs

echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

# ---- conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepfake311   # <-- change if your env differs

# ---- runtime env ----
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Expose only GPUs granted by Slurm (torchrun will further pick per-rank)
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

# ---- paths / args ----
SCRIPT="/home/infres/ziyliu-24/FakeParts2/StyleTrans/AnyV2V/AnyV2VPipline.py"
VIDEOS_CSV="/projects/hi-paris/DeepFakeDataset/DeepFake_V2/10k_real_video_captions_ziyi.csv"
OUT_DIR="/projects/hi-paris/DeepFakeDataset/DeepFake_V2/V2V/AnyV2V"
ANYV2V_ROOT="/home/infres/ziyliu-24/FakeParts2/StyleTrans/AnyV2V"

# How many total videos to process across ALL ranks (the script will shard)
NUM_SAMPLES=5000

# number of processes = number of GPUs on this node
NP=${SLURM_GPUS_ON_NODE:-2}

# ---- quick sanity ----
nvidia-smi -L || true
srun --gres=gpu:"${NP}" python - <<'PY'
import os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

# make sure the dest exists
mkdir -p "/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition/Change_of_style/AnyV2V/fake_videos"

# expand the glob safely; avoid errors when empty
shopt -s nullglob
mp4s=(/projects/hi-paris/DeepFakeDataset/DeepFake_V2/V2V/AnyV2V/*/color_change/*.mp4)
if ((${#mp4s[@]})); then
  cp -n "${mp4s[@]}" "/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition/Change_of_style/AnyV2V/fake_videos/"
fi

# ---- launch ----
srun --gres=gpu:"${NP}" torchrun \
  --standalone \
  --nproc_per_node="${NP}" \
  "${SCRIPT}" \
  --csv "${VIDEOS_CSV}" \
  --output_root "${OUT_DIR}" \
  --anyv2v_root "${ANYV2V_ROOT}" \
  --num "${NUM_SAMPLES}" \
  --repeat    # drop this flag if you want to skip already-done items

EXIT_CODE=$?
echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"
