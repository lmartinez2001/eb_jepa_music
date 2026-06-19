import matplotlib.pyplot as plt
import numpy as np

COCO_17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

KEYPOINT_COLORS = plt.cm.tab20(np.linspace(0, 1, 17))


def draw_keypoints2d(
    ax, keypoints, edges=COCO_17_EDGES, keypoint_colors=KEYPOINT_COLORS
):
    points = np.asarray(keypoints)
    valid = np.isfinite(points).all(axis=-1) & (points[:, 0] >= 0) & (points[:, 1] >= 0)
    colors = np.asarray(keypoint_colors)

    for i, j in edges:
        if i < len(points) and j < len(points) and valid[i] and valid[j]:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                color=colors[j % len(colors)],
                linewidth=1.6,
                alpha=0.9,
            )

    for joint_idx, point in enumerate(points):
        if not valid[joint_idx]:
            continue
        ax.scatter(
            point[0],
            point[1],
            s=20,
            c=[colors[joint_idx % len(colors)]],
            edgecolors="black",
            linewidths=0.45,
            alpha=0.98,
            zorder=3,
        )


def visualize_aistpp_batch(
    batch, max_samples=4, max_frames=6, overlay_keypoints=True, show_audio=True
):
    images = batch["images"].detach().cpu()
    keypoints2d = batch.get("keypoints2d")
    if keypoints2d is not None:
        keypoints2d = keypoints2d.detach().cpu().numpy()
    audio_spectrograms = batch.get("audio_spectrograms")
    if audio_spectrograms is not None:
        audio_spectrograms = audio_spectrograms.detach().cpu().numpy()
    audio_chunks = batch.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunks = audio_chunks.detach().cpu().numpy()

    batch_size, num_frames, _, height, width = images.shape
    nrows = min(max_samples, batch_size)
    ncols = min(max_frames, num_frames)
    has_audio = show_audio and (
        audio_spectrograms is not None or audio_chunks is not None
    )
    plot_rows = nrows * (2 if has_audio else 1)
    frame_positions = np.linspace(0, num_frames - 1, ncols).round().astype(int)

    fig, axes = plt.subplots(
        plot_rows,
        ncols,
        figsize=(3.2 * ncols, (3.15 if has_audio else 2.35) * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row in range(nrows):
        image_row = row * (2 if has_audio else 1)
        audio_row = image_row + 1

        for col, frame_pos in enumerate(frame_positions):
            ax = axes[image_row, col]
            frame = images[row, frame_pos].permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(frame)

            if overlay_keypoints and keypoints2d is not None:
                draw_keypoints2d(ax, keypoints2d[row, frame_pos])

            timestamp = float(batch["frame_timestamps"][row, frame_pos])
            frame_id = int(batch["frame_ids"][row, frame_pos])
            ax.set_title(f"t={timestamp:.1f}s / f={frame_id}", fontsize=9)
            ax.axis("off")

            if has_audio:
                audio_ax = axes[audio_row, col]
                if audio_spectrograms is not None:
                    spec = audio_spectrograms[row, frame_pos]
                    audio_ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
                else:
                    waveform = audio_chunks[row, frame_pos]
                    audio_ax.plot(waveform, color="black", linewidth=0.6)
                    max_amplitude = max(float(np.max(np.abs(waveform))), 1e-6)
                    audio_ax.set_ylim(-1.05 * max_amplitude, 1.05 * max_amplitude)
                audio_ax.set_xticks([])
                audio_ax.set_yticks([])

        axes[image_row, 0].set_ylabel(
            f"#{batch['meta']['dataset_index'][row]} "
            f"{batch['meta']['video_name'][row]}\n"
            f"view={batch['meta']['view'][row]}",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=54,
        )
        if has_audio:
            axes[audio_row, 0].set_ylabel(
                "mel dB" if audio_spectrograms is not None else "wave",
                fontsize=9,
                rotation=0,
                ha="right",
                va="center",
                labelpad=28,
            )

    return fig