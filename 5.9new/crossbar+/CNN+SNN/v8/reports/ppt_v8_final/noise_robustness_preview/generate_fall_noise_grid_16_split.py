from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\ppt_v8_final")
SOURCE_IMAGE = ROOT / "recognition_case_gallery" / "frames" / "Fall_07_TP.jpg"
OUT_DIR = ROOT / "noise_robustness_preview"

LEVELS = [i / 15 for i in range(16)]
NOISE_LABELS = {
    "gaosi": "gaussian_noise_grid_16points.png",
    "possion": "poisson_noise_grid_16points.png",
    "salt": "salt_noise_grid_16points.png",
}

WRAP_COLS = 4
TILE_W = 92
TILE_H = 62
PADDING = 10
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


def render_single_grid(noise_type: str, output_name: str) -> Path:
    base = resize_frame(SOURCE_IMAGE)
    rows = (len(LEVELS) + WRAP_COLS - 1) // WRAP_COLS
    width = PADDING + WRAP_COLS * (TILE_W + PADDING)
    height = PADDING + rows * (TILE_H + PADDING)
    canvas = Image.new("RGB", (width, height), CANVAS_FILL)
    draw = ImageDraw.Draw(canvas)

    for level_idx, level in enumerate(LEVELS):
        row = level_idx // WRAP_COLS
        col = level_idx % WRAP_COLS
        x0 = PADDING + col * (TILE_W + PADDING)
        y0 = PADDING + row * (TILE_H + PADDING)
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

    output_path = OUT_DIR / output_name
    canvas.save(output_path, quality=95)
    return output_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for noise_type, output_name in NOISE_LABELS.items():
        print(render_single_grid(noise_type, output_name))


if __name__ == "__main__":
    main()
