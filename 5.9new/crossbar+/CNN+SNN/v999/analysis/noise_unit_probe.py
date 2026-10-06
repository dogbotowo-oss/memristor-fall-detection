"""Direct probe of the crossbar readout-noise implementation.

For a fixed synthetic window and fixed seed, measures the RMS perturbation
|output(scale) - output(0)| actually injected at different readout_noise_scale
values. Verifies (a) the scale override reaches the noise path and
(b) quantifies the effective sigma as a fraction of full scale.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np

CODE = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v999\code\automatic_fall_detector.py")

spec = importlib.util.spec_from_file_location("afd_probe", CODE)
mod = importlib.util.module_from_spec(spec)
sys.modules["afd_probe"] = mod
spec.loader.exec_module(mod)

candidates = [
    (name, fn)
    for name, fn in inspect.getmembers(mod, inspect.isfunction)
    if "readout_noise_scale" in inspect.signature(fn).parameters
]
print("functions with readout_noise_scale:", [n for n, _ in candidates])

name, fn = candidates[0]
print("using:", name, list(inspect.signature(fn).parameters))

rng = np.random.default_rng(0)
window = rng.random((16, 64, 64), dtype=np.float32)
hrs = np.array([0.08, 0.10, 0.12, 0.15], dtype=np.float32)
lrs = np.array([0.70, 0.78, 0.86, 0.92], dtype=np.float32)

clean = fn(window, hrs, lrs, 12345, readout_noise_scale=0.0)
for scale in (0.15, 0.25, 0.60, 0.90, 1.50, 2.00):
    out = fn(window, hrs, lrs, 12345, readout_noise_scale=scale)
    diff = out.astype(np.float64) - clean.astype(np.float64)
    rms = float(np.sqrt((diff ** 2).mean()))
    p99 = float(np.percentile(np.abs(diff), 99))
    print(f"scale={scale:4.2f}  RMS pert={rms:.5f} ({rms*100:.2f}% FS)  p99|d|={p99:.5f}")

# isolate: 0.15 vs 0.90 (same rng stream; only readout sigma differs)
a = fn(window, hrs, lrs, 12345, readout_noise_scale=0.15)
b = fn(window, hrs, lrs, 12345, readout_noise_scale=0.90)
d = b.astype(np.float64) - a.astype(np.float64)
rms = float(np.sqrt((d ** 2).mean()))
print(f"0.15-vs-0.90 (readout-only): RMS={rms:.5f} ({rms*100:.2f}% FS)")
