from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"F:\12.18-2")
OUTPUT_ROOT = ROOT / "噪声-数据集" / "model_input_64x64"

SOURCES = {
    "train": ROOT / "zenodo_falldb_video_split" / "train",
    "val": ROOT / "zenodo_falldb_video_split" / "val",
    "test": ROOT / "gmdcsa_subject4_test" / "Subject 4",
}

NOISE_TYPES = ["gaosi", "possion", "salt"]
LEVELS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".m4v", ".mkv"}
IMAGE_SIZE = 64


def level_tag(level: float) -> str:
    return f"level_{int(round(level * 1000)):03d}"


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) % (2**32)
    return value


def poisson_peak(level_index: int) -> float | None:
    if level_index == 0:
        return None
    t = level_index / (len(LEVELS) - 1)
    return float(220.0 * ((6.0 / 220.0) ** t))


def apply_noise(gray01: np.ndarray, noise_type: str, level: float, level_index: int, rng: np.random.Generator) -> np.ndarray:
    if level_index == 0:
        return gray01
    if noise_type == "gaosi":
        sigma = 0.30 * level
        return np.clip(gray01 + rng.normal(0.0, sigma, size=gray01.shape).astype(np.float32), 0.0, 1.0)
    if noise_type == "possion":
        peak = float(poisson_peak(level_index) or 220.0)
        return np.clip(rng.poisson(np.clip(gray01, 0.0, 1.0) * peak).astype(np.float32) / peak, 0.0, 1.0)
    if noise_type == "salt":
        out = gray01.copy()
        mask = rng.random(gray01.shape) < level
        salt_mask = rng.random(gray01.shape) < 0.5
        out[mask & salt_mask] = 1.0
        out[mask & ~salt_mask] = 0.0
        return out
    raise ValueError(f"Unsupported noise type: {noise_type}")


def list_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def write_noisy_video(src: Path, dst: Path, noise_type: str, level: float, level_index: int) -> dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return {"status": "skipped", "frames": "", "fps": "", "width": IMAGE_SIZE, "height": IMAGE_SIZE}

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMAGE_SIZE, IMAGE_SIZE), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open output writer: {dst}")

    rng = np.random.default_rng(stable_seed(noise_type, level_index, src))
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            gray01 = small.astype(np.float32) / 255.0
            noisy = apply_noise(gray01, noise_type, level, level_index, rng)
            out_gray = (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
            writer.write(out_bgr)
            frames += 1
    finally:
        cap.release()
        writer.release()
    return {"status": "written", "frames": frames, "fps": fps, "width": IMAGE_SIZE, "height": IMAGE_SIZE}


def write_readme() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    text = f"""# 64x64 噪声测试数据集

本目录是独立保存的噪声副本，不覆盖原始数据。

生成口径：

- 按 `v8 final` 模型实际输入尺寸保存：`64x64`
- 保留 `train / val / test` 和 `ADL / Fall` 分类结构
- train 来源：`{SOURCES["train"]}`
- val 来源：`{SOURCES["val"]}`
- test 来源：`{SOURCES["test"]}`

噪声类型：

- `gaosi`: 高斯噪声，`sigma = 0.30 * level`
- `possion`: 泊松噪声，使用 photon peak 控制强度
- `salt`: 椒盐噪声，`level` 表示像素污染比例

强度等级：

`0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%`

目录示例：

`model_input_64x64/gaosi/level_125/train/ADL/*.mp4`
"""
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-type", choices=NOISE_TYPES + ["all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_readme()
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    fields = [
        "noise_type",
        "level_tag",
        "level_percent",
        "split",
        "source_path",
        "output_path",
        "status",
        "frames",
        "fps",
        "width",
        "height",
        "poisson_peak",
    ]
    existing_rows: list[dict[str, object]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    rows = existing_rows
    noise_types = NOISE_TYPES if args.noise_type == "all" else [args.noise_type]
    for noise_type in noise_types:
        for level_index, level in enumerate(LEVELS):
            tag = level_tag(level)
            for split, src_root in SOURCES.items():
                for src in list_videos(src_root):
                    rel = src.relative_to(src_root)
                    dst = OUTPUT_ROOT / noise_type / tag / split / rel.with_suffix(".mp4")
                    result = write_noisy_video(src, dst, noise_type, level, level_index)
                    row = {
                        "noise_type": noise_type,
                        "level_tag": tag,
                        "level_percent": level * 100.0,
                        "split": split,
                        "source_path": str(src),
                        "output_path": str(dst),
                        "poisson_peak": poisson_peak(level_index) if noise_type == "possion" else "",
                        **result,
                    }
                    rows.append(row)
                    print(f"{noise_type} {tag} {split} | {rel} | {row['status']}", flush=True)
                    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)

    summary = {
        "output_root": str(OUTPUT_ROOT),
        "image_size": IMAGE_SIZE,
        "noise_types": noise_types,
        "levels": LEVELS,
        "sources": {key: str(value) for key, value in SOURCES.items()},
        "manifest": str(manifest_path),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
