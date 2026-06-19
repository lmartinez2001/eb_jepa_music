from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio.transforms as audio_transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class AISTPPClipDataset(Dataset):
    """PyTorch wrapper around the Hugging Face AIST++ dataset.

    Each item is one clip with evenly sampled frames and aligned annotations.

    Returns:
        images: [T, C, H, W] float tensor in [0, 1]
        keypoints2d: [T, J, 2] float tensor, scaled to image_size
        keypoints3d: [T, J, 3] float tensor
        frame_ids: [T] long tensor
        frame_timestamps: [T] float tensor
        audio_chunks: [T, A] float tensor, if include_audio_chunks=True
        audio_spectrograms: [T, n_mels, S] float tensor, if include_audio_spectrograms=True
        meta: dict of non-tensor sample metadata
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 12,
        image_size: Optional[Tuple[int, int]] = (256, 144),
        include_audio_chunks: bool = True,
        include_audio_spectrograms: bool = True,
        audio_sample_rate: int = 48000,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 256,
        audio_n_mels: int = 64,
        dataset_name: str = "haiphamcse/aistpp",
    ):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.include_audio_chunks = include_audio_chunks
        self.include_audio_spectrograms = include_audio_spectrograms
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.audio_n_mels = audio_n_mels
        self.dataset_name = dataset_name
        self.hf_dataset = load_dataset(dataset_name, split=split)
        self.audio_to_spectrogram = audio_transforms.MelSpectrogram(
            sample_rate=audio_sample_rate,
            n_fft=audio_n_fft,
            hop_length=audio_hop_length,
            n_mels=audio_n_mels,
            power=2.0,
        )
        self.audio_to_db = audio_transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.hf_dataset)

    def _frame_indices(self, n_frames: int) -> np.ndarray:
        if n_frames <= 0:
            raise ValueError("Sample has no frames")
        return np.linspace(0, n_frames - 1, self.num_frames).round().astype(np.int64)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.image_size is not None and image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _scale_keypoints2d(
        self, keypoints: np.ndarray, original_size: Tuple[int, int]
    ) -> np.ndarray:
        if self.image_size is None:
            return keypoints

        original_w, original_h = original_size
        target_w, target_h = self.image_size
        scaled = keypoints.copy()
        scaled[..., 0] *= target_w / original_w
        scaled[..., 1] *= target_h / original_h
        return scaled

    def _audio_chunks_to_spectrograms(self, audio_chunks: torch.Tensor) -> torch.Tensor:
        spectrograms = self.audio_to_spectrogram(audio_chunks)
        return self.audio_to_db(spectrograms)

    def __getitem__(self, idx: int):
        sample = self.hf_dataset[idx]
        images = sample["images"]
        frame_idx = self._frame_indices(len(images))

        original_size = images[0].size
        frames = torch.stack(
            [self._image_to_tensor(images[i]) for i in frame_idx], dim=0
        )

        keypoints2d = np.asarray(sample["keypoints2d"], dtype=np.float32)[frame_idx]
        keypoints2d = self._scale_keypoints2d(keypoints2d, original_size)

        item = {
            "images": frames,
            "keypoints2d": torch.from_numpy(keypoints2d),
            "keypoints3d": torch.as_tensor(
                np.asarray(sample["keypoints3d"], dtype=np.float32)[frame_idx]
            ),
            "frame_ids": torch.as_tensor(
                np.asarray(sample["frame_ids"], dtype=np.int64)[frame_idx]
            ),
            "frame_timestamps": torch.as_tensor(
                np.asarray(sample["frame_timestamps"], dtype=np.float32)[frame_idx]
            ),
            "meta": {
                "dataset_index": idx,
                "video_name": sample["video_name"],
                "annotation_name": sample["annotation_name"],
                "view": sample["view"],
                "video_basename": sample["video_basename"],
                "stride": sample["stride"],
                "fps_sampled": sample["fps_sampled"],
                "audio_sr": sample["audio_sr"],
                "original_size": original_size,
                "image_size": self.image_size or original_size,
            },
        }

        if self.include_audio_chunks or self.include_audio_spectrograms:
            audio_chunks = torch.as_tensor(
                np.asarray(sample["audio_chunks"], dtype=np.float32)[frame_idx]
            )
            if self.include_audio_chunks:
                item["audio_chunks"] = audio_chunks
            if self.include_audio_spectrograms:
                item["audio_spectrograms"] = self._audio_chunks_to_spectrograms(
                    audio_chunks
                )

        return item


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from eb_jepa.datasets.aistpp.utils import visualize_aistpp_batch

    dataset = AISTPPClipDataset(
        split="train",
        num_frames=12,
        image_size=(256, 144),
        include_audio_chunks=True,
        include_audio_spectrograms=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(4, len(dataset)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    batch = next(iter(loader))

    shapes = {
        "images": tuple(batch["images"].shape),
        "keypoints2d": tuple(batch["keypoints2d"].shape),
        "keypoints3d": tuple(batch["keypoints3d"].shape),
        "frame_ids": tuple(batch["frame_ids"].shape),
        "frame_timestamps": tuple(batch["frame_timestamps"].shape),
    }
    if "audio_chunks" in batch:
        shapes["audio_chunks"] = tuple(batch["audio_chunks"].shape)
    if "audio_spectrograms" in batch:
        shapes["audio_spectrograms"] = tuple(batch["audio_spectrograms"].shape)
    print(shapes)

    fig = visualize_aistpp_batch(batch, max_samples=4, max_frames=6)
    fig.savefig("aistpp_batch.png", dpi=150)
    print("Saved visualization to aistpp_batch.png")