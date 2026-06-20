#!/bin/bash
#SBATCH --job-name=music_jepa
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /lustre/work/vivatech-4dudes/edugelay/eb_jepa_music/env.sh

cd "$EBJEPA_REPO"

python music/main.py \
    --optim.epochs=200 \
    --meta.save_dir=/lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa \
    --data.num_workers=0 \
    --data.batch_size=128 \
    --logging.log_wandb=true
    