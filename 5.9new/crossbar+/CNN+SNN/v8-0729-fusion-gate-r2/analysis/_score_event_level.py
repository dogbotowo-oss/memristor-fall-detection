import importlib.util, json, sys
import numpy as np
from pathlib import Path

root_tag = sys.argv[1] if len(sys.argv) > 1 else "batch_val_v88"
EXP = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
ROOT = EXP / "reports" / root_tag

sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", str(EXP / "code" / "run_refined_gate_r2.py"))
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

tp = tn = fp = fn = 0
per_split = {}
for split in ("subject3", "zenodo_val"):
    summary = json.loads((ROOT / split / "summary.json").read_text(encoding="utf-8"))
    stp = stn = sfp = sfn = 0
    for row in summary["results"]:
        if split == "subject3":
            rel = str(Path(row["video_path"]).relative_to(s3dir))
            tf = s3_labels.get(rel)
        else:
            tf = str(Path(row["video_path"]).relative_to(DROOT / "zenodo_falldb_video_split" / "val")).startswith("Fall")
        if tf is None:
            continue
        pf = row.get("process_segment") is not None
        if tf and pf: stp += 1
        elif tf and not pf: sfn += 1
        elif (not tf) and pf: sfp += 1
        else: stn += 1
    tp += stp; tn += stn; fp += sfp; fn += sfn
    sspec = stn / max(stn+sfp,1); srec = stp / max(stp+sfn,1)
    sacc = (stp+stn)/max(stp+stn+sfp+sfn,1)
    per_split[split] = (stp, stn, sfp, sfn, sacc, 0.5*(sspec+srec), sspec, srec)

for split, (a,b,c,d,acc,bal,spec,rec) in per_split.items():
    print(f"{split:10s} TP/TN/FP/FN={a}/{b}/{c}/{d} acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} rec={rec:.4f}")
spec = tn / max(tn+fp,1); rec = tp / max(tp+fn,1)
acc = (tp+tn)/(tp+tn+fp+fn); bal = 0.5*(spec+rec)
print(f"COMBINED   TP/TN/FP/FN={tp}/{tn}/{fp}/{fn} acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} rec={rec:.4f}")
