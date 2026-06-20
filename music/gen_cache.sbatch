#!/bin/bash
#SBATCH --job-name=gen_cache
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /lustre/work/vivatech-4dudes/edugelay/eb_jepa_music/env.sh

cd "$EBJEPA_REPO"

python -m music.datasets.decoder_dataset \
    --checkpoint /lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa/dev_2026-06-20_15-22/exp_seed2025/epoch_2.pth.tar \
    --out-dir /lustre/work/vivatech-4dudes/edugelay/datasets/decoder_cache \
    --num-workers 0 \
    --use-amp
