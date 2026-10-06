from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"F:\12.18-2")
V15_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v15"
CODE_PATH = V15_DIR / "code" / "automatic_fall_detector.py"
OUTPUT_JSON = V15_DIR / "reports" / "v15_subtype_dryrun_summary.json"

GMDCSA_ROOT = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
TRAIN_DIRS = [
    GMDCSA_ROOT / "Subject 1",
    GMDCSA_ROOT / "Subject 2",
    ROOT / "zenodo_falldb_video_split" / "train",
]
VAL_DIRS = [
    GMDCSA_ROOT / "Subject 3",
    ROOT / "zenodo_falldb_video_split" / "val",
]


def load_module():
    spec = importlib.util.spec_from_file_location("automatic_fall_detector_v15_dryrun", CODE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {CODE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def count_subtypes(mod, dataset, hard_threshold: float) -> dict[str, object]:
    all_normal: list[int] = []
    hard_only: list[int] = []
    targeted_default: list[int] = []
    targeted_default_hard: list[int] = []
    targeted_ids = set(mod.resolve_adl_subtype_ids(list(mod.DEFAULT_TARGETED_ADL_SUBTYPE_NAMES)))

    for sample in dataset.samples:
        if int(sample["label_id"]) != mod.LABEL_NAME_TO_ID["normal"]:
            continue
        subtype_id = int(sample.get("adl_subtype_id", mod.ADL_SUBTYPE_IGNORE_ID))
        hard_score = float(sample.get("hard_negative_score", 0.0))
        if subtype_id < 0:
            continue
        all_normal.append(subtype_id)
        if subtype_id in targeted_ids:
            targeted_default.append(subtype_id)
        if hard_score >= hard_threshold:
            hard_only.append(subtype_id)
            if subtype_id in targeted_ids:
                targeted_default_hard.append(subtype_id)

    return {
        "all_normal_counts": mod.adl_subtype_count_dict(all_normal),
        "hard_only_counts": mod.adl_subtype_count_dict(hard_only),
        "targeted_default_counts": mod.adl_subtype_count_dict(targeted_default),
        "targeted_default_hard_counts": mod.adl_subtype_count_dict(targeted_default_hard),
        "hard_threshold": float(hard_threshold),
        "normal_windows_total": int(len(all_normal)),
        "normal_windows_hard_only": int(len(hard_only)),
    }


def build_dataset(mod, windows: np.ndarray, labels: np.ndarray, base_seed: int):
    return mod.CombinedFallDataset(
        video_windows=windows,
        video_labels=labels,
        auxiliary_records=[],
        window_size=16,
        image_size=64,
        use_crossbar=False,
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
        hrs_values=np.asarray([0.1], dtype=np.float32),
        lrs_values=np.asarray([0.9], dtype=np.float32),
        device_dynamics=None,
        crossbar_readout_noise_scale=0.0,
        augment=False,
        base_seed=base_seed,
    )


def main() -> None:
    mod = load_module()

    train_windows, train_labels, train_summary = mod.load_external_labeled_windows(
        TRAIN_DIRS,
        image_size=64,
        window_size=16,
        stride=16,
        max_videos=0,
        fall_name_pattern="fall",
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
        adl_context_path_keyword="zenodo_falldb_video_split/train/adl",
        adl_context_min_percentile=0.95,
        adl_context_radius_windows=0,
        adl_context_extra_copies=1,
    )
    val_windows, val_labels, val_summary = mod.load_external_labeled_windows(
        VAL_DIRS,
        image_size=64,
        window_size=16,
        stride=16,
        max_videos=0,
        fall_name_pattern="fall",
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
        adl_context_path_keyword="",
        adl_context_min_percentile=0.95,
        adl_context_radius_windows=0,
        adl_context_extra_copies=0,
    )

    train_dataset = build_dataset(mod, train_windows, train_labels, base_seed=1234)
    val_dataset = build_dataset(mod, val_windows, val_labels, base_seed=5678)
    hard_threshold = 0.22

    payload = {
        "version": "v15_subtype_dryrun",
        "data_split": {
            "train_dirs": [str(path) for path in TRAIN_DIRS],
            "val_dirs": [str(path) for path in VAL_DIRS],
            "external_labeled_stride": 16,
            "window_size": 16,
            "image_size": 64,
            "adl_context_path_keyword": "zenodo_falldb_video_split/train/adl",
            "adl_context_min_percentile": 0.95,
            "adl_context_extra_copies": 1,
            "strict_rule": "Subject1-2 + Zenodo train for train; Subject3 + Zenodo val for validation only",
        },
        "targeted_default_subtypes": list(mod.DEFAULT_TARGETED_ADL_SUBTYPE_NAMES),
        "train": {
            "label_counts": mod.class_count_dict(train_labels),
            "dataset_summary": train_summary,
            "adl_subtypes": count_subtypes(mod, train_dataset, hard_threshold=hard_threshold),
            "adl_subtype_thresholds": train_dataset.adl_subtype_thresholds,
        },
        "val": {
            "label_counts": mod.class_count_dict(val_labels),
            "dataset_summary": val_summary,
            "adl_subtypes": count_subtypes(mod, val_dataset, hard_threshold=hard_threshold),
            "adl_subtype_thresholds": val_dataset.adl_subtype_thresholds,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
