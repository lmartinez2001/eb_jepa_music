# Music-Conditioned Dance JEPA

Music-conditioned Joint Embedding Predictive Architecture (JEPA) for dance motion generation on AIST++.

## Architecture

```
Audio (WAV)
  └─ AudioEncoder (MuQ-large, frozen)
       └─ frame-level sequence [B, T, 1024]
                                          ┐
Pose clip [B, F, 25, 3]                  │
  └─ DSTformer encoder                   ├─ MusicTransformerPredictor
       └─ z_t [B, 512]  ────────────────►│       └─ z_pred [B, H, 512]
                                          ┘              │
Target clips [B, H, F, 25, 3]                            │  VICReg
  └─ DSTformer encoder (stop-grad)                       │  (std + cov)
       └─ z_targets [B, H, 512]  ◄───────────────────────┘
```

### Components

**DSTformer** (`models/encoder.py`) — keypoint encoder. Takes a window of 25-joint 3D poses `[B, F, J, 3]` and returns a `cls_state` embedding `[B, 512]` via the last-frame CLS token.

**AudioEncoder** (`models/audio_encoder.py`) — wraps frozen MuQ-large (1024-d hidden). Returns the full frame-level sequence `[B, T, 1024]`; no pooling here.

**MusicTransformerPredictor** (`models/transformer_predictor.py`) — autoregressive latent predictor. Music and state tokens are interleaved into a causal sequence `[m₀, z_t, m₁, z_{t+1}, …]`. An `AttentionPooling` head (1024 → 512) pools each MuQ sequence into a music token before interleaving. Supports teacher-forcing during training and `generate()` for autoregressive inference.

**VICReg** — prevents representational collapse via a per-dimension std hinge `max(0, 1−std(z))` and an off-diagonal covariance penalty.

**MotionDecoder** (`models/motion_decoder.py`) — DiT-based flow-matching decoder. Conditioned on a JEPA latent `z`, generates a normalised pose clip `[B, F, 25, 3]` via Euler integration of a learned velocity field.

---

## Dataset

AIST++ v3 stored as a HuggingFace Arrow dataset. Raw 5-second clips (`_w0 / _w1 / …`) are stitched into variable-length sequences at load time. Each training sample contains:

- `keypoints` — `[window + pred_horizon × horizon, 25, 3]` (z-score normalised)
- `music` — `[pred_horizon, chunk_samples]` — one audio chunk per future prediction step

Key config parameters (`cfgs/train.yaml`):

| param | default | meaning |
|---|---|---|
| `data.window` | 60 | frames per pose window (= encoder context) |
| `data.horizon` | 60 | frame gap between consecutive prediction steps |
| `data.pred_horizon` | 3 | number of future steps to predict |
| `model.music_encoder.chunk_frames` | 120 | audio frames per music chunk |

---

## Training

### 1 — JEPA (joint embedding + predictor)

```bash
sbatch music/train.sbatch
```

Config: `cfgs/train.yaml`. Key overrides accepted as `--key=value`:

```bash
# example: longer run, custom save dir
sbatch music/train.sbatch   # edits to train.sbatch propagate automatically
```

Logs to Weights & Biases (`logging.log_wandb: true`). Checkpoints:

- `<save_dir>/latest.pth.tar` — always the most recent epoch (used for resumption)
- `<run_folder>/best.pth.tar` — best `val/pred_loss` within this run

Monitor collapse via `collapse/mean_std_norm` and `collapse/effective_rank` in W&B. With VICReg, `mean_std_norm` should stay above ~0.8.

---

### 2 — Generate decoder cache

Once the JEPA checkpoint is satisfactory, generate the shard cache that the decoder trains on. Each JEPA sample produces `pred_horizon` decoder samples (flattened).

```bash
# Edit gen_cache.sbatch to point at the desired JEPA checkpoint, then:
sbatch music/gen_cache.sbatch
```

This writes to `--out-dir` (default `datasets/decoder_cache/`):

```
decoder_cache/
  train/  shard_000000.pt  …  manifest.pt
  val/    shard_000000.pt  …  manifest.pt
  normalizer.pt        ← mean/std needed to unnormalise at inference
  source_config.yaml
  metadata.json
```

---

### 3 — Motion decoder

```bash
sbatch music/train_decoder.sbatch
```

Config: `cfgs/train_decoder.yaml`. `data.cache_dir` must point at the directory generated above. The decoder is a DiT flow-matching model trained with a straight-path CFM objective.

---

## Inference

```bash
python music/infer.py \
    --audio        /path/to/song.wav \
    --jepa-ckpt    checkpoints/music_jepa/<run>/best.pth.tar \
    --decoder-ckpt checkpoints/music_decoder/<run>/best.pth.tar \
    --cache-dir    datasets/decoder_cache \
    --out          output/dance.npz
```

Optional flags:

| flag | default | meaning |
|---|---|---|
| `--init-pose` | `None` | NPY file `[F, J, 3]` (unnormalised) to seed z_t |
| `--num-steps` | 50 | Euler steps for flow-matching ODE |

Output: `dance.npz` with key `poses` of shape `[pred_horizon × window, 25, 3]` in real-world coordinates.
