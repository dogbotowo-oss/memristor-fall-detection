import importlib.util, json, sys
import numpy as np
from pathlib import Path
sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\code\run_refined_gate_r2.py")
r2 = importlib.util.module_from_spec(spec); sys.modules["r2"] = r2; spec.loader.exec_module(r2)
base = r2.base
ROOT = Path(r"E:\rray\12.18-2")
G = ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
s3dir = G / "Subject 3"
w, l, s = base.load_external_labeled_windows([s3dir], image_size=64, window_size=16, stride=16, max_videos=0, fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20)
fall_id = base.LABEL_NAME_TO_ID["fall"]
label_map = {}
off = 0
for row in s["videos"]:
    n = int(row["num_windows"])
    any_fall = bool((l[off:off+n] == fall_id).any())
    rel = str(Path(row["video_path"]).relative_to(s3dir))
    label_map[rel] = any_fall
    off += n
print("true fall videos:", sum(label_map.values()), "adl:", len(label_map) - sum(label_map.values()))
summary = json.loads(Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\reports\batch_val_v88\subject3\summary.json").read_text(encoding="utf-8"))
tp = tn = fp = fn = miss = 0
for row in summary["results"]:
    rel = str(Path(row["video_path"]).relative_to(s3dir))
    if rel not in label_map:
        miss += 1
        continue
    tf = label_map[rel]
    pf = row.get("process_segment") is not None
    if tf and pf: tp += 1
    elif tf and not pf: fn += 1
    elif (not tf) and pf: fp += 1
    else: tn += 1
spec = tn / max(tn+fp,1)
rec = tp / max(tp+fn,1)
acc = (tp+tn)/max(tp+tn+fp+fn,1)
print(f"Subject3 used={tp+tn+fp+fn} missing={miss} TP/TN/FP/FN={tp}/{tn}/{fp}/{fn}")
print(f"acc={acc:.4f} bal={0.5*(spec+rec):.4f} spec={spec:.4f} rec={rec:.4f}")
