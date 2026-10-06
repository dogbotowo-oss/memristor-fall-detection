from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("F:/12.18-2")
OPT_NAME = "\u4f18\u53167.3"
EXP = ROOT / "5.9new" / "crossbar+" / OPT_NAME
CODE_DIR = EXP / "code"
IMAGE_DIR = EXP / "image"
REPORT_DIR = EXP / "reports"
MODEL_PATH = EXP / "models" / "automatic_fall_event_detector_opt73_hardneg_shape.pth"
DETECTOR_PATH = CODE_DIR / "automatic_fall_detector.py"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test"
PREVIOUS_BEST = ROOT / "5.9new" / "crossbar+" / "teacher_crossbar_device_hardneg" / "reports" / "recommended_threshold.json"


def load_detector():
    spec = importlib.util.spec_from_file_location("opt73_detector", DETECTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import detector from {DETECTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_row(name: str, path: Path) -> dict:
    data = read_json(path)
    result = data["subject4_result"]
    fall_correct = int(result["fall_correct"])
    false_alarms = int(result["false_alarms"])
    precision = fall_correct / max(fall_correct + false_alarms, 1)
    return {
        "name": name,
        "threshold": data["recommended_threshold"],
        "accuracy": float(result["video_accuracy"]),
        "adl_specificity": float(result["adl_specificity"]),
        "fall_recall": float(result["fall_recall"]),
        "precision": precision,
        "balanced_accuracy": float(result["balanced_accuracy"]),
        "false_alarms": false_alarms,
        "missed_falls": int(result["missed_falls"]),
        "adl_correct": int(result["adl_correct"]),
        "fall_correct": fall_correct,
    }


def save_result_comparison() -> dict:
    opt_path = REPORT_DIR / "recommended_threshold.json"
    rows = [
        result_row("Previous B+D", PREVIOUS_BEST),
        result_row("Opt 7.3 hardneg", opt_path),
    ]
    delta = {key: rows[1][key] - rows[0][key] for key in ["accuracy", "adl_specificity", "fall_recall", "precision", "balanced_accuracy"]}
    summary = {"rows": rows, "delta": delta}
    (REPORT_DIR / "opt73_effect_comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.text(0.02, 0.93, "Optimization 7.3 Effect on Subject4", fontsize=24, fontweight="bold", color="#0f2744")
    ax.text(0.02, 0.875, "Only this experiment changes the hard-negative ADL definition; other settings keep the B+D recipe.", fontsize=12.5, color="#526174")

    headers = ["Experiment", "Thr p/m/t", "Acc", "ADL", "Fall", "Precision", "Bal.", "FP/FN"]
    x = [0.02, 0.245, 0.405, 0.515, 0.625, 0.735, 0.86, 0.965]
    y0 = 0.73
    row_h = 0.12
    ax.add_patch(plt.Rectangle((0.015, y0 - 0.02), 0.97, 0.07, color="#e6edf5", transform=ax.transAxes))
    for i, h in enumerate(headers):
            ax.text(x[i], y0, h, fontsize=11.5, fontweight="bold", color="#0f2744", transform=ax.transAxes, va="center", ha="left" if i < 7 else "right")
    for r, row in enumerate(rows):
        y = y0 - 0.09 - r * row_h
        if r == 1:
            ax.add_patch(plt.Rectangle((0.015, y - 0.04), 0.97, 0.09, color="#d9efe5", transform=ax.transAxes))
        thr = row["threshold"]
        values = [
            row["name"],
            f"{thr['fall_prob_threshold']:.2f}/{thr['fall_margin_threshold']:.2f}/{thr['min_fall_duration_sec']:.2f}s",
            pct(row["accuracy"]),
            pct(row["adl_specificity"]),
            pct(row["fall_recall"]),
            pct(row["precision"]),
            pct(row["balanced_accuracy"]),
            f"{row['false_alarms']}/{row['missed_falls']}",
        ]
        for i, value in enumerate(values):
            ax.text(x[i], y, value, fontsize=12.5, fontweight="bold" if r == 1 or i == 0 else "normal", color="#0f2744", transform=ax.transAxes, va="center", ha="left" if i < 7 else "right")

    ax.text(0.02, 0.38, "Net gain from 7.3:", fontsize=15, fontweight="bold", color="#0f2744", transform=ax.transAxes)
    gain_lines = [
        f"Acc {delta['accuracy'] * 100:+.2f} points",
        f"ADL specificity {delta['adl_specificity'] * 100:+.2f} points",
        f"Fall recall {delta['fall_recall'] * 100:+.2f} points",
        f"Precision {delta['precision'] * 100:+.2f} points",
        f"Balanced accuracy {delta['balanced_accuracy'] * 100:+.2f} points",
    ]
    for i, line in enumerate(gain_lines):
        ax.text(0.04, 0.32 - i * 0.055, line, fontsize=13.5, color="#1a7856", transform=ax.transAxes)
    ax.text(0.50, 0.38, "Interpretation:", fontsize=15, fontweight="bold", color="#0f2744", transform=ax.transAxes)
    ax.text(0.52, 0.32, "Shape-change hard negatives reduce ADL confusion while also recovering one extra Fall video.", fontsize=13.2, color="#526174", transform=ax.transAxes)
    ax.text(0.52, 0.265, "The chosen threshold becomes slightly more permissive in probability (0.75) but stricter in duration (0.75s).", fontsize=13.2, color="#526174", transform=ax.transAxes)
    out = IMAGE_DIR / "opt73_effect_comparison_ppt.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return summary


def save_device_curve_figure(detector) -> None:
    dynamics = detector.load_device_dynamics(ROOT / "data")
    steps = 64
    epsc = detector.resample_curve_to_frames(np.asarray(dynamics["epsc_curve"], dtype=np.float32), steps)
    ppf = detector.resample_curve_to_frames(np.asarray(dynamics["ppf_curve"], dtype=np.float32), steps)
    ltp = detector.resample_curve_to_frames(np.asarray(dynamics["ltp_curve"], dtype=np.float32), steps)
    ltd = detector.resample_curve_to_frames(np.asarray(dynamics["ltd_curve"], dtype=np.float32), steps)
    dyn = 0.55 + 0.35 * epsc + 0.20 * ppf + 0.15 * ltp + 0.10 * ltd
    row_sigma = 0.006 * 0.25 * dyn
    read_sigma = 0.004 * 0.25 * dyn

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), dpi=150, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#d7dee8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    x = np.arange(steps)
    axes[0].plot(x, epsc, label="EPSC", linewidth=2.4)
    axes[0].plot(x, ppf, label="PPF", linewidth=2.4)
    axes[0].plot(x, ltp, label="LTP", linewidth=2.4)
    axes[0].plot(x, ltd, label="LTD", linewidth=2.4)
    axes[0].set_ylabel("Normalized device response")
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    axes[0].set_title("Device Dynamics Curves Used by Crossbar Readout", loc="left", fontsize=18, fontweight="bold", color="#0f2744")
    axes[1].plot(x, dyn, label="dynamic factor", color="#6d28d9", linewidth=2.8)
    axes[1].plot(x, row_sigma * 100.0, label="row/column gain sigma (%)", color="#2563eb", linewidth=2.4)
    axes[1].plot(x, read_sigma * 100.0, label="read noise sigma (%)", color="#d97706", linewidth=2.4)
    axes[1].set_ylabel("Readout modulation")
    axes[1].set_xlabel("Resampled frame index in a window")
    axes[1].legend(ncol=3, frameon=False, loc="upper right")
    axes[1].text(0.01, 0.06, "Formula: dynamic_factor = 0.55 + 0.35*EPSC + 0.20*PPF + 0.15*LTP + 0.10*LTD; scale = 0.25", transform=axes[1].transAxes, fontsize=11, color="#526174")
    fig.savefig(IMAGE_DIR / "device_dynamics_readout_explain_ppt.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_index = max(0, int(frame_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return np.zeros((160, 220, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_case_lists() -> tuple[list[dict], list[dict], list[dict]]:
    summary = read_json(REPORT_DIR / "sub4_calibrated_batch_summary.json")
    false_alarms, missed_falls, correct_falls = [], [], []
    for item in summary["results"]:
        report = read_json(Path(item["report_json"]))
        video_path = Path(report["video_path"])
        is_adl = "ADL" in {part.upper() for part in video_path.parts} or "ADL" in video_path.as_posix()
        is_fall = "Fall" in {part for part in video_path.parts} or "Fall" in video_path.as_posix()
        process = report.get("process_segment")
        key = report.get("key_frame") or {}
        peak_score = float((process or {}).get("peak_score", key.get("score", 0.0)))
        row = {
            "video_path": str(video_path),
            "report_json": str(Path(item["report_json"])),
            "process_segment": process,
            "key_frame": key,
            "peak_score": peak_score,
            "fps": float(item.get("fps", report.get("fps", 30.0)) or 30.0),
            "prediction_counts": report.get("prediction_counts", {}),
        }
        if is_adl and process is not None:
            false_alarms.append(row)
        if is_fall and process is None:
            missed_falls.append(row)
        if is_fall and process is not None:
            correct_falls.append(row)
    false_alarms.sort(key=lambda x: x["peak_score"], reverse=True)
    missed_falls.sort(key=lambda x: x["peak_score"], reverse=True)
    correct_falls.sort(key=lambda x: x["peak_score"], reverse=True)
    return false_alarms, missed_falls, correct_falls


def run_inference(detector, model, config: dict, video_path: Path, scale: float, threshold: dict) -> dict:
    frames, fps = detector.read_video_frames(
        video_path,
        image_size=int(config.get("image_size", 128)),
        use_pixel_human=bool(config.get("use_pixel_human", False)),
        use_silhouette_human=bool(config.get("use_silhouette_human", False)),
        pixel_grid_size=int(config.get("pixel_grid_size", 20)),
    )
    hrs, lrs = detector.load_conductance_pair(ROOT / "data")
    dynamics = detector.load_device_dynamics(ROOT / "data")
    return detector.predict_video_frames(
        model=model,
        frames=frames,
        fps=fps,
        window_size=int(config.get("window_size", 16)),
        stride=int(config.get("infer_stride", 4)),
        use_crossbar=bool(config.get("use_crossbar", True)),
        hrs_values=hrs,
        lrs_values=lrs,
        device_dynamics=dynamics,
        crossbar_readout_noise_scale=scale,
        device=detector.get_device(False),
        batch_size=16,
        fall_prob_threshold=float(threshold["fall_prob_threshold"]),
        fall_margin_threshold=float(threshold["fall_margin_threshold"]),
        min_fall_duration_sec=float(threshold["min_fall_duration_sec"]),
    )


def save_probability_toggle_figure(detector, false_alarms: list[dict], correct_falls: list[dict]) -> None:
    threshold = read_json(REPORT_DIR / "recommended_threshold.json")["recommended_threshold"]
    device = detector.get_device(False)
    model, config = detector.load_model_checkpoint(MODEL_PATH, device)
    model.eval()
    case = false_alarms[0] if false_alarms else correct_falls[0]
    video_path = Path(case["video_path"])
    clean = run_inference(detector, model, config, video_path, scale=0.0, threshold=threshold)
    noisy = run_inference(detector, model, config, video_path, scale=0.25, threshold=threshold)
    fps = float(noisy["fps"])
    t = np.arange(int(noisy["num_frames"])) / max(fps, 1e-6)
    peak = int((case.get("key_frame") or {}).get("frame_index", np.argmax(noisy["frame_probabilities"][:, 3])))
    window = int(config.get("window_size", 16))
    start = max(0, peak - window // 2)
    end = min(len(t) - 1, start + window - 1)
    frame = read_frame(video_path, peak)

    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.8], height_ratios=[1.0, 0.36], hspace=0.22, wspace=0.18)
    ax_img = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])
    ax_img.imshow(frame)
    ax_img.axis("off")
    ax_img.set_title(f"Selected window: {video_path.parent.name}/{video_path.name}\npeak frame {peak}", loc="left", fontsize=13, fontweight="bold", color="#0f2744")
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#d7dee8", linewidth=0.8)
    ax.plot(t, clean["frame_probabilities"][:, 3], label="fall prob, readout scale 0.00", color="#1a7856", linewidth=2.4)
    ax.plot(t, noisy["frame_probabilities"][:, 3], label="fall prob, readout scale 0.25", color="#d97706", linewidth=2.4)
    ax.plot(t, noisy["frame_probabilities"][:, 0], label="normal prob, scale 0.25", color="#2563eb", alpha=0.65, linewidth=1.8)
    ax.axhline(float(threshold["fall_prob_threshold"]), color="#334155", linestyle="--", linewidth=1.3, label="fall threshold")
    ax.axvspan(start / fps, end / fps, color="#fde68a", alpha=0.45, label="same model window")
    ax.set_title("Same Video Window: Crossbar Readout Noise Off vs On", loc="left", fontsize=17, fontweight="bold", color="#0f2744")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper right")

    clean_window = clean["frame_probabilities"][start : end + 1, 3]
    noisy_window = noisy["frame_probabilities"][start : end + 1, 3]
    labels = ["Mean fall prob", "Peak fall prob"]
    clean_vals = [float(clean_window.mean()), float(clean_window.max())]
    noisy_vals = [float(noisy_window.mean()), float(noisy_window.max())]
    xpos = np.arange(len(labels))
    ax_bar.set_facecolor("#ffffff")
    ax_bar.bar(xpos - 0.18, clean_vals, width=0.36, label="scale 0.00", color="#1a7856")
    ax_bar.bar(xpos + 0.18, noisy_vals, width=0.36, label="scale 0.25", color="#d97706")
    ax_bar.set_xticks(xpos, labels)
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(True, axis="y", color="#d7dee8", linewidth=0.8)
    ax_bar.legend(frameon=False, ncol=2, loc="upper right")
    ax_bar.text(0.02, 0.88, "Readout noise is applied after HRS/LRS mapping as row/column gain drift plus read noise.", transform=ax_bar.transAxes, fontsize=11.5, color="#526174")
    fig.savefig(IMAGE_DIR / "probability_noise_toggle_ppt.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_error_analysis_figure(false_alarms: list[dict], missed_falls: list[dict]) -> None:
    selected_false = false_alarms[:4]
    selected_miss = missed_falls[:4]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10.5), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes.ravel():
        ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.80, bottom=0.09, wspace=0.20, hspace=0.65)
    fig.suptitle("Visual Error Analysis after Optimization 7.3", fontsize=22, fontweight="bold", color="#0f2744", x=0.02, y=0.97, ha="left")
    rows = [("ADL false alarms", selected_false, "#b45309"), ("Fall missed cases", selected_miss, "#b91c1c")]
    for row_idx, (row_name, cases, color) in enumerate(rows):
        fig.text(0.04, 0.83 if row_idx == 0 else 0.43, row_name, fontsize=15, fontweight="bold", color=color)
        for col in range(4):
            ax = axes[row_idx, col]
            if col >= len(cases):
                ax.text(0.5, 0.5, "No case", ha="center", va="center", fontsize=13, color="#64748b")
                continue
            case = cases[col]
            video_path = Path(case["video_path"])
            key = case.get("key_frame") or {}
            process = case.get("process_segment") or {}
            frame_idx = int(process.get("start_frame", key.get("frame_index", 0)))
            frame = read_frame(video_path, frame_idx)
            ax.imshow(frame)
            ax.axis("off")
            title = f"{video_path.parent.name}/{video_path.name}\n"
            if process:
                title += f"segment {process.get('start_time_sec', 0):.2f}-{process.get('end_time_sec', 0):.2f}s, peak={case['peak_score']:.3f}"
            else:
                title += f"no detected segment, key={int(key.get('frame_index', 0))}, score={case['peak_score']:.3f}"
            ax.set_title(title, fontsize=10.5, color="#0f2744", loc="left")
    fig.text(0.02, 0.025, "False alarms are ADL videos with a detected fall segment; missed cases are Fall videos without a calibrated fall segment.", fontsize=11.5, color="#526174")
    fig.savefig(IMAGE_DIR / "error_analysis_false_alarm_miss_ppt.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), CODE_DIR / "generate_opt73_explainability.py")
    detector = load_detector()
    comparison = save_result_comparison()
    save_device_curve_figure(detector)
    false_alarms, missed_falls, correct_falls = load_case_lists()
    save_probability_toggle_figure(detector, false_alarms, correct_falls)
    save_error_analysis_figure(false_alarms, missed_falls)
    summary = {
        "comparison": comparison,
        "false_alarm_count": len(false_alarms),
        "missed_fall_count": len(missed_falls),
        "figures": [
            str(IMAGE_DIR / "opt73_effect_comparison_ppt.png"),
            str(IMAGE_DIR / "device_dynamics_readout_explain_ppt.png"),
            str(IMAGE_DIR / "probability_noise_toggle_ppt.png"),
            str(IMAGE_DIR / "error_analysis_false_alarm_miss_ppt.png"),
        ],
    }
    (REPORT_DIR / "opt73_explainability_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
