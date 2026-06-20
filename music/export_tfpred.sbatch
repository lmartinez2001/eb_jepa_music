#!/bin/bash
#SBATCH --job-name=export_tfpred
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/work/vivatech-4dudes/dpham/working/eb_jepa_music/music/%x_%j.out
#SBATCH --error=/lustre/work/vivatech-4dudes/dpham/working/eb_jepa_music/music/%x_%j.out

source /lustre/work/vivatech-4dudes/dpham/working/eb_jepa_music/env.sh
cd "$EBJEPA_REPO"
export PYTHONPATH="$EBJEPA_REPO:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- synced env.sh no longer activates the venv / sets lib paths; do it here ---
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
_av="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/av.libs"
[ -d "$_av" ] && export LD_LIBRARY_PATH="$_av${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
_nv="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
[ -d "$_nv" ] && export LD_LIBRARY_PATH="$_nv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CKPT=/lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa/dev_2026-06-20_13-53/exp_seed2025/epoch_10.pth.tar
CFG=/lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa/dev_2026-06-20_13-53/exp_seed2025/config.yaml
CACHE=/lustre/work/vivatech-4dudes/dpham/working/datasets/music_decoder_cache_tfpred

echo "[check] python: $(which python)"
echo "[smoke] validating model path"
python music/smoke_tfpred.py || { echo "SMOKE_FAIL"; exit 1; }

echo "[export] generating tfpred rot6d cache (generate h=0)"
python music/datasets/decoder_dataset.py \
    --out-dir "$CACHE" --checkpoint "$CKPT" --config "$CFG" \
    --shard-size 4096 --use-amp
