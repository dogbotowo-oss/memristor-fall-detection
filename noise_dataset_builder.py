from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"F:\12.18-2")
OUTPUT_ROOT = ROOT / "噪声-数据集"

SOURCES = {
    "train": ROOT / "zenodo_falldb_video_split" / "train",
    "val": ROOT / "zenodo_falldb_video_split" / "val",
    "test": ROOT / "gmdcsa_subject4_test" / "Subject 4",
}

NOISE_TYPES = ["gaosi", "possion", "salt"]
LEVELS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".m4v", ".mkv"}


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
    max_peak = 220.0
    min_peak = 6.0
    return float(max_peak * ((min_peak / max_peak) ** t))


def apply_noise(frame_bgr: np.ndarray, noise_type: str, level: float, level_index: int, rng: np.random.Generator) -> np.ndarray:
    if level_index == 0:
        return frame_bgr

    x = frame_bgr.astype(np.float32) / 255.0

    if noise_type == "gaosi":
        sigma = 0.30 * level
        y = np.clip(x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    if noise_type == "possion":
        peak = float(poisson_peak(level_index) or 220.0)
        y = rng.poisson(np.clip(x, 0.0, 1.0) * peak).astype(np.float32) / peak
        return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

    if noise_type == "salt":
        y = x.copy()
        mask = rng.random(x.shape[:2]) < level
        salt_mask = rng.random(x.shape[:2]) < 0.5
        y[mask & salt_mask, :] = 1.0
        y[mask & ~salt_mask, :] = 0.0
        return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

    raise ValueError(f"Unsupported noise type: {noise_type}")


def list_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def write_noisy_video(src: Path, dst: Path, noise_type: str, level: float, level_index: int) -> dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return {"status": "skipped", "frames": None, "fps": None, "width": None, "height": None}

    if level_index == 0:
        shutil.copy2(src, dst)
        return {"status": "copied_clean", "frames": None, "fps": None, "width": None, "height": None}

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Unable to read video size: {src}")

    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
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
            writer.write(apply_noise(frame, noise_type, level, level_index, rng))
            frames += 1
    finally:
        cap.release()
        writer.release()

    return {"status": "written", "frames": frames, "fps": fps, "width": width, "height": height}


def write_readme() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    text = f"""# 噪声测试数据集

本目录是独立生成的噪声副本，不覆盖原始训练集、验证集或测试集。

源数据：

- train: `{SOURCES["train"]}`
- val: `{SOURCES["val"]}`
- test: `{SOURCES["test"]}`

噪声类型：

- `gaosi`: 高斯噪声，`sigma = 0.30 * level`
- `possion`: 泊松噪声，使用 photon peak 控制强度，peak 越低噪声越强
- `salt`: 椒盐噪声，`level` 表示像素污染比例

强度等级：

`0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%`

目录结构示例：

`噪声-数据集/gaosi/level_125/train/ADL/*.mp4`

说明：

- `level_000` 为无噪声干净副本。
- 原始数据不被修改。
- `manifest.csv` 记录每个输出视频的来源、噪声类型、强度和生成状态。
"""
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
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

    rows: list[dict[str, object]] = []
    for noise_type in NOISE_TYPES:
        for level_index, level in enumerate(LEVELS):
            tag = level_tag(level)
            for split, src_root in SOURCES.items():
                videos = list_videos(src_root)
                for src in videos:
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
                    print(
                        f"{noise_type} {tag} {split} | {rel} | {row['status']}",
                        flush=True,
                    )
                    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)

    summary = {
        "output_root": str(OUTPUT_ROOT),
        "noise_types": NOISE_TYPES,
        "levels": LEVELS,
        "sources": {key: str(value) for key, value in SOURCES.items()},
        "manifest": str(manifest_path),
        "total_rows": len(rows),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
