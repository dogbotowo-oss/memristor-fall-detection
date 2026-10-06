import importlib.util, sys
from pathlib import Path
sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\code\run_refined_gate_r2.py")
r2 = importlib.util.module_from_spec(spec); sys.modules["r2"] = r2; spec.loader.exec_module(r2)
base = r2.base
ROOT = Path(r"E:\rray\12.18-2")
G = ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
w, l, s = base.load_external_labeled_windows([G / "Subject 3", ROOT / "zenodo_falldb_video_split" / "val"], image_size=64, window_size=16, stride=16, max_videos=0, fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20)
print("rows:", len(s["videos"]), "windows:", len(w))
total = 0
for row in s["videos"]:
    n = int(row.get("num_windows", 0))
    total += n
    print(repr(row.get("video_name")), repr(row.get("assigned_label")), n)
print("sum num_windows:", total)
