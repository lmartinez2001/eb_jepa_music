#!/bin/bash
#SBATCH --job-name=dec_tfpred
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/work/vivatech-4dudes/dpham/working/eb_jepa_music/music/%x_%j.out
#SBATCH --error=/lustre/work/vivatech-4dudes/dpham/working/eb_jepa_music/music/%x_%j.out

source /lustre/work/vivatech-4dudes/dpham/working/eb_jepa_music/env.sh
cd "$EBJEPA_REPO"
export PYTHONPATH="$EBJEPA_REPO:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="$(tr -d "[:space:]" < /lustre/work/vivatech-4dudes/dpham/working/wand.txt)"

CACHE=/lustre/work/vivatech-4dudes/dpham/working/datasets/music_decoder_cache_tfpred
FOLDER=/lustre/work/vivatech-4dudes/dpham/checkpoints/music_decoder_tfpred

python music/train_decoder.py \
    --data.cache_dir="$CACHE" \
    --meta.model_folder="$FOLDER" \
    --model.decoder.num_frames=60 \
    --model.decoder.coord_dim=6 \
    --model.decoder.cond_dim=512 \
    --model.decoder.num_joints=25 \
    --data.batch_size=64 \
    --optim.epochs=300 \
    --logging.save_every=25 \
    --logging.val_every=20 \
    --logging.val_ode_steps=20 \
    --logging.val_max_batches=20 \
    --logging.log_wandb=true
