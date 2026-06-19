"""MuQ-MuLan audio<->text similarity on an aistpp_v2 clip.

Loads one audio sample from haiphamcse/aistpp_v2 and compares two ways of
embedding it against the 10 AIST dance-genre text prompts:

  1. whole sample  -> one embedding -> similarity with each genre
  2. short windows -> a 2s window centered on each of the 150 SMPL frames,
     embed each, similarity per window, then mean over windows.

MuLan needs 24kHz mono; the dataset audio is 48kHz stereo, so we downmix +
resample. Audio is read from raw bytes (Audio(decode=False)) to avoid torchcodec.

Run:
  python demo.py                       # first sample (a gBR/Break clip)
  python demo.py --name gPO_sBM_cAll_d10_mPO0_ch01
  python demo.py --index 42
"""
import io
import argparse
import itertools

import numpy as np
import soundfile as sf
import librosa
import torch
from datasets import load_dataset, Audio
from muq import MuQMuLan

REPO = "haiphamcse/aistpp_v2"
MULAN_SR = 24000          # MuLan input rate
WINDOW_SEC = 2.0          # centered window length

# 10 AIST genres: code -> (name, text prompt for MuLan)
GENRES = [
    ("gBR", "Break",            "breakdance music"),
    ("gPO", "Pop",              "pop dance music"),
    ("gLO", "Lock",             "locking funk dance music"),
    ("gMH", "Middle Hip-hop",   "middle school hip-hop dance music"),
    ("gLH", "LA Hip-hop",       "LA style hip-hop dance music"),
    ("gHO", "House",            "house dance music"),
    ("gWA", "Waack",            "waacking disco dance music"),
    ("gKR", "Krump",            "krump dance music"),
    ("gJS", "Street Jazz",      "street jazz dance music"),
    ("gJB", "Ballet Jazz",      "ballet jazz dance music"),
]
PROMPTS = [p for _, _, p in GENRES]
NAMES = [n for _, n, _ in GENRES]


def load_sample(name, index):
    """Stream one row; return (name, fps, mono24k waveform float32)."""
    ds = load_dataset(REPO, split="train", streaming=True).cast_column(
        "audio", Audio(decode=False))
    if name is not None:
        row = next(r for r in ds if r["name"] == name)
    else:
        row = next(itertools.islice(ds, index, index + 1))

    audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32",
                        always_2d=True)          # (N, 2) @ 48000
    mono = audio.mean(axis=1)                    # downmix -> (N,)
    mono24k = librosa.resample(mono, orig_sr=sr, target_sr=MULAN_SR)
    return row["name"], int(row["fps"]), mono24k.astype(np.float32)


def make_windows(wav, fps):
    """A WINDOW_SEC window centered on each frame -> (n_frames, win_samples)."""
    spf = MULAN_SR // fps                         # samples per frame (800)
    n_frames = len(wav) // spf                     # 150
    win = int(round(WINDOW_SEC * MULAN_SR))        # 48000
    half = win // 2
    padded = np.pad(wav, (half, half))             # so every frame gets a full window
    starts = (np.arange(n_frames) + 0.5) * spf     # center on each frame
    return np.stack([padded[int(s):int(s) + win] for s in starts]).astype(np.float32)


@torch.no_grad()
def embed_audio(mulan, wav_batch, device, bs=32):
    """wav_batch: (M, T) float32 -> (M, D) audio embeddings (batched)."""
    out = []
    for i in range(0, len(wav_batch), bs):
        wt = torch.from_numpy(wav_batch[i:i + bs]).to(device)
        out.append(mulan(wavs=wt))
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="sequence name (overrides --index)")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    name, fps, wav = load_sample(args.name, args.index)
    true_genre = name[:3]
    print(f"sample: {name}  (true genre: {true_genre})  fps={fps}  "
          f"audio={len(wav)} samples @ {MULAN_SR}Hz")

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(args.device).eval()

    with torch.no_grad():
        text_embeds = mulan(texts=PROMPTS)                       # (10, D)

    # 1) whole sample
    whole_embed = embed_audio(mulan, wav[None, :], args.device)  # (1, D)
    sim_whole = mulan.calc_similarity(whole_embed, text_embeds)[0]  # (10,)

    # 2) short centered windows, one per frame
    windows = make_windows(wav, fps)                             # (150, 48000)
    win_embeds = embed_audio(mulan, windows, args.device)        # (150, D)
    sim_win = mulan.calc_similarity(win_embeds, text_embeds)     # (150, 10)
    sim_win_mean = sim_win.mean(dim=0)                           # (10,)

    sw = sim_whole.float().cpu().numpy()
    swm = sim_win_mean.float().cpu().numpy()
    print(f"\nwindows: {windows.shape[0]} x {WINDOW_SEC}s centered on each frame\n")
    print(f"{'genre':<16}{'whole':>10}{'win-mean':>10}")
    print("-" * 36)
    for i, gname in enumerate(NAMES):
        mark = "  <- true" if GENRES[i][0] == true_genre else ""
        print(f"{gname:<16}{sw[i]:>10.4f}{swm[i]:>10.4f}{mark}")
    print(f"\ntop-1  whole    : {NAMES[int(sw.argmax())]}")
    print(f"top-1  win-mean : {NAMES[int(swm.argmax())]}")


if __name__ == "__main__":
    main()
