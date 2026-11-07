# Data shipped with this repository

Everything in this folder is small, redistributable device or pose data that the code needs in
order to run. Video datasets are **not** included (see below).

## Device measurements (`data/`)

All device files are Agilent B1500A exports; the loader in `automatic_fall_detector.py`
(`load_conductance_values`) reads the `DataValue, <t>, <I>` rows and keeps positive currents.

| file | measurement | read bias | samples | duration | used for |
|---|---|---|---|---|---|
| `HRS.csv` | high-resistance state, continuous read | 0.1 V | 10,001 @ 1 s | 1.0 × 10⁴ s | conductance distribution for device-aware encoding / weight quantisation |
| `LRS.csv` | low-resistance state, continuous read | 0.1 V | 2,223 @ 1 s | 2.2 × 10³ s | same as above |
| `EPSC.csv` | single 2 V / 1 ms pulse, 0.2 V read | 0.2 V | 1 ms window | — | excitatory post-synaptic current term |
| `LTP-1ms-阶梯付 [(1) ; 7_28_2026 7_59_16 PM].csv` | potentiation staircase | — | — | — | long-term plasticity term |
| `LTD-1ms-阶梯 [(1) ; 7_28_2026 7_53_06 PM].csv` | depression staircase | — | — | — | long-term plasticity term |
| `PPF-0.001-60-60.xls` | paired-pulse facilitation | — | — | — | short-term plasticity term |

`HRS.csv` and `LRS.csv` are **independent retention measurements**, not the long-duration
retention trace used for the stability figure (10⁵ s at 0.01 V read, plotted separately in the
manuscript). They are only used to obtain the two readout-current distributions.

## Trained weights (`models/`)

Four leave-one-subject-out checkpoints (seed 11, full device model) used by
`5.9new/crossbar+/CNN+SNN/s30_pose/scripts/repro_v1001_protocol.py`:

```
models/automatic_fall_event_detector_c1_s34_data1_loso_ts{1,2,3,4}_s11.pth
```

## Pose cache (`pose_cache/`)

8-D MediaPipe BlazePose features per frame, cached as `.npz` per video (392 files, 26.5 MB).
The checkpoints store an absolute `pose_cache_dir` from the original machine, so evaluation
must override it, otherwise every pose feature is silently zero and all videos are classified
as normal activity. Pass `pose_cache/` (this folder) as the cache directory, e.g.

```bash
python 5.9new/crossbar+/CNN+SNN/s30_pose/scripts/repro_v1001_protocol.py \
    --pose-cache-dir pose_cache
```

## Not included

| item | why | where to get it |
|---|---|---|
| GMDCSA-24 v2.1 videos (Subjects 1–4) | dataset licence / size | https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos |
| Zenodo fall-detection split (18 validation videos) | dataset licence / size | Zenodo record cited in the manuscript |
| `I_V (100cycle).csv` (4.5 MB) | only needed for the I–V plots | available from the authors on request |
| `fall_video_frame_labels.csv` | not present on the original machine | regenerate from the dataset folder names |
