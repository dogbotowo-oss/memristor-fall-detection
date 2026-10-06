import json
import numpy as np
from pathlib import Path

ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\reports\batch_val_v88")
entries = []
for summary_path, true_map in [
    (ROOT / "subject3" / "summary.json", None),
    (ROOT / "zenodo_val" / "summary.json", None),
]:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in s["results"]:
        report = json.loads(Path(row["report_json"]).read_text(encoding="utf-8"))
        peak = float(report["key_frame"]["score"])
        vp = Path(row["video_path"])
        entries.append({"video": vp.name, "dir": summary_path.parent.name, "path": str(vp), "peak": peak})

# true labels: subject3 from any-fall windows (recompute), zenodo from subdir
import importlib.util, sys
sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\code\run_refined_gate_r2.py")
r2 = importlib.util.module_from_spec(spec); sys.modules["r2"] = r2; spec.loader.exec_module(r2)
base = r2.base
DROOT = Path(r"E:\rray\12.18-2")
G = DROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
s3dir = G / "Subject 3"
w, l, s = base.load_external_labeled_windows([s3dir], image_size=64, window_size=16, stride=16, max_videos=0, fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20)
fall_id = base.LABEL_NAME_TO_ID["fall"]
s3_labels = {}
off = 0
for row in s["videos"]:
    n = int(row["num_windows"])
    rel = str(Path(row["video_path"]).relative_to(s3dir))
    s3_labels[rel] = bool((l[off:off+n] == fall_id).any())
    off += n

for e in entries:
    if e["dir"] == "subject3":
        rel = str(Path(e["path"]).relative_to(s3dir))
        e["true_fall"] = s3_labels[rel]
    else:
        e["true_fall"] = str(Path(e["path"]).relative_to(DROOT / "zenodo_falldb_video_split" / "val")).startswith("Fall")

print(f"videos={len(entries)} fall={sum(e['true_fall'] for e in entries)} adl={sum(1 for e in entries if not e['true_fall'])}")
print("thr | TP TN FP FN | acc bal spec rec")
best = None
for thr in np.arange(0.40, 0.901, 0.05):
    tp = tn = fp = fn = 0
    for e in entries:
        pf = e["peak"] >= thr
        tf = e["true_fall"]
        if tf and pf: tp += 1
        elif tf and not pf: fn += 1
        elif (not tf) and pf: fp += 1
        else: tn += 1
    spec = tn / max(tn+fp,1); rec = tp / max(tp+fn,1)
    acc = (tp+tn)/len(entries); bal = 0.5*(spec+rec)
    print(f"{thr:.2f} | {tp:2d} {tn:2d} {fp:2d} {fn:2d} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} rec={rec:.4f}")
    if best is None or bal > best[1]:
        best = (float(thr), bal, acc, spec, rec)
print(f"best: thr={best[0]:.2f} bal={best[1]:.4f} acc={best[2]:.4f} spec={best[3]:.4f} rec={best[4]:.4f}")
