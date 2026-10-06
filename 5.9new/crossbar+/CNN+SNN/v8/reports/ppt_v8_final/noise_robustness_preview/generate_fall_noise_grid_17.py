from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\ppt_v8_final")
SOURCE_IMAGE = ROOT / "recognition_case_gallery" / "frames" / "Fall_07_TP.jpg"
OUT_DIR = ROOT / "noise_robustness_preview"
OUTPUT_IMAGE = OUT_DIR / "fall_noise_grid_17points.png"

# 17 points: 0%, 6.25%, ..., 100%
LEVELS = [i / 16 for i in range(17)]
NOISE_TYPES = ("gaosi", "possion", "salt")
WRAP_COLS = 6

TILE_W = 92
TILE_H = 62
PADDING = 10
SECTION_GAP = 18
CARD_RADIUS = 8
CARD_FILL = (248, 249, 252)
CARD_BORDER = (222, 226, 235)
CANVAS_FILL = (244, 246, 250)


def poisson_peak(level_index: int) -> float | None:
    if level_index == 0:
        return None
    t = level_index / (len(LEVELS) - 1)
    max_peak = 220.0
    min_peak = 6.0
    return float(max_peak * ((min_peak / max_peak) ** t))


def stable_seed(noise_type: str, level_index: int) -> int:
    base = {"gaosi": 1103, "possion": 2207, "salt": 3301}[noise_type]
    return base + level_index * 97


def apply_image_noise(image: np.ndarray, noise_type: str, level: float, level_index: int) -> np.ndarray:
    if level_index == 0:
        return image.astype(np.float32, copy=True)

    rng = np.random.default_rng(stable_seed(noise_type, level_index))
    x = image.astype(np.float32, copy=True)

    if noise_type == "gaosi":
        sigma = 0.30 * float(level)
        out = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    if noise_type == "possion":
        peak = float(poisson_peak(level_index) or 220.0)
        out = rng.poisson(np.clip(x, 0.0, 1.0) * peak).astype(np.float32) / peak
        return np.clip(out, 0.0, 1.0)

    if noise_type == "salt":
        out = x.copy()
        mask = rng.random(x.shape) < float(level)
        salt_mask = rng.random(x.shape) < 0.5
        out[mask & salt_mask] = 1.0
        out[mask & ~salt_mask] = 0.0
        return np.clip(out, 0.0, 1.0)

    raise ValueError(f"Unsupported noise_type: {noise_type}")


def resize_frame(frame_path: Path) -> np.ndarray:
    frame = Image.open(frame_path).convert("RGB").resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)
    return np.asarray(frame, dtype=np.float32) / 255.0


def render_grid() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = resize_frame(SOURCE_IMAGE)
    cols = WRAP_COLS
    rows_per_noise = (len(LEVELS) + WRAP_COLS - 1) // WRAP_COLS
    rows = len(NOISE_TYPES) * rows_per_noise

    width = PADDING + cols * (TILE_W + PADDING)
    height = PADDING + rows * (TILE_H + PADDING) + (len(NOISE_TYPES) - 1) * SECTION_GAP
    canvas = Image.new("RGB", (width, height), CANVAS_FILL)
    draw = ImageDraw.Draw(canvas)

    for noise_idx, noise_type in enumerate(NOISE_TYPES):
        block_row_offset = noise_idx * rows_per_noise
        block_y_offset = noise_idx * SECTION_GAP
        for level_idx, level in enumerate(LEVELS):
            row_in_block = level_idx // WRAP_COLS
            col_in_row = level_idx % WRAP_COLS
            items_in_row = min(WRAP_COLS, len(LEVELS) - row_in_block * WRAP_COLS)
            row_width = items_in_row * TILE_W + (items_in_row - 1) * PADDING
            max_width = WRAP_COLS * TILE_W + (WRAP_COLS - 1) * PADDING
            x_start = PADDING + max(0, (max_width - row_width) // 2)

            x0 = x_start + col_in_row * (TILE_W + PADDING)
            y0 = PADDING + (block_row_offset + row_in_block) * (TILE_H + PADDING) + block_y_offset
            x1 = x0 + TILE_W
            y1 = y0 + TILE_H

            draw.rounded_rectangle(
                (x0 - 1, y0 - 1, x1 + 1, y1 + 1),
                radius=CARD_RADIUS,
                fill=CARD_FILL,
                outline=CARD_BORDER,
                width=1,
            )

            noisy = apply_image_noise(base, noise_type, level, level_idx)
            tile = Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8))
            canvas.paste(tile, (x0, y0))

    canvas.save(OUTPUT_IMAGE, quality=95)
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    render_grid()
