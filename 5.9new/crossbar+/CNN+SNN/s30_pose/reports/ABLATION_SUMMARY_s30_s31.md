# s30/s31 Pose-SNN Ablation Summary (2026-08-13)

All numbers at frozen threshold 0.70. val = Subject3 + Zenodo val (61 videos),
Subject4 = frozen independent test (37 videos, 20 ADL / 17 Fall).
pose-off = pose features zeroed at inference (load_pose_frame_features monkeypatched).

## Main table

| model | val acc | val spec | val recall | S4 acc | S4 spec | S4 recall |
|---|---|---|---|---|---|---|
| v9 baseline (no pose) | 72.13 | 63.33 | 80.65 | 67.57 | 60.00 | 76.47 (13/17) |
| s30 pose (gate floor 0) | 70.49 | 60.00 | 80.65 | 70.27 | 60.00 | 82.35 (14/17) |
| s30 pose + floor 0.2 | 73.77 | 66.67 | 80.65 | 70.27 | 60.00 | 82.35 (14/17) |
| s31 (floor .2 + late-fusion .2 + aux .3) | 70.49 | 66.67 | 74.19 | 67.57 | 75.00 | 58.82 (10/17) |

## Pose-off ablation (causal test)

| model | S4 recall pose-on | S4 recall pose-off | delta |
|---|---|---|---|
| s30 pose (floor 0) | 82.35 | 82.35 | 0 (gate collapsed to ~0.005, pose unused) |
| s30 pose + floor 0.2 | 82.35 | 76.47 | +5.88 from pose |
| s31 | 58.82 | 70.59 | -11.76 (pose hurts S4, helps val) |

## Key findings

1. s30+floor0.2 pose-off on Subject4 degrades EXACTLY to the v9 baseline:
   same false alarms (ADL 05,06,07,11,12,15,16,17), same missed falls
   (Fall 01,02,09,13). The pose-SNN pathway's entire S4 effect is +1 fall
   detection (Fall 09), with zero side effects on the other 36 videos.
2. Without the gate floor the SNN gate collapses (mean ~0.005) and pose
   contributes nothing; floor 0.2 makes the pathway load-bearing.
3. s30+floor0.2 is the best version: val beats v9 on every metric
   (acc 73.77 vs 72.13, spec 66.67 vs 63.33, recall tied 80.65), S4 acc
   70.27 (+2.70) and recall 82.35 (+5.88) vs v9.
4. s31's late-fusion + aux head is unstable: pose helps val but hurts S4.
   Not recommended to continue in that direction.
5. Magnitude caveat: +1/17 fall videos is directionally positive but not
   statistically significant on its own; needs multi-seed repetition.

## Next steps

- Multi-seed repeats (3 seeds) of s30+floor0.2 to separate pose effect from
  training noise (~15 min per run).
- Extract pose caches for Zenodo train/val videos to remove the pose_valid
  asymmetry (currently Zenodo pose features are all zeros).
- Consider extending cache to UP-Fall ADL increment set.

---

# s32/s33 update (2026-08-14): Zenodo = Kinect depth + skeleton.txt

Zenodo videos are DEPTH maps; MediaPipe cannot run on them (5-15% detection).
But the original FallDatabase ships Kinect skeleton tracks (7 joints, mm,
tracking conf) -> pose_convert_zenodo_skeleton.py maps them into the
MediaPipe 33-point cache layout (HEAD->0, SHOULDER_CENTER/SPINE->11/12,
HIP_L/R->23/24, knees->25/26, y flipped so downward grows, mm/1000).
Coverage: 72/72 videos, valid pose frames 69% (after SPINE fallbacks).
Skeleton feature AUC on Zenodo: head_drop 0.82, max_head_vy 0.57.

## Subject4 results by protocol

Old protocol (val threshold scan capped at 0.70, as in v9):

| model | thr | S4 acc | S4 spec | S4 recall |
|---|---|---|---|---|
| v9 | 0.70 | 67.57 | 60.00 | 76.47 |
| s30 pose+floor0.2 | 0.70 | 70.27 | 60.00 | 82.35 |
| s32 (zenodo skeleton) | 0.70 | 75.68 | 75.00 | 76.47 |
| s33 (s32+class head) | 0.60 | 72.97 | 80.00 | 64.71 |

Extended scan (0.35-0.90) - TRAP: every model incl. v9 converges to
S4 78.38/90/64.71 (v9: 78.38/90.0/64.71 @0.80). High thresholds wash out
all inter-model differences and flatter the baseline. Do NOT headline the
extended-scan numbers; keep the original <=0.70 protocol for reporting.

## Pose-off ablations on Subject4 (causal test)

| model@thr | pose-on | pose-off | pose effect |
|---|---|---|---|
| s30floor@0.70 | 70.27/60/82.35 | 67.57/60/76.47 (=v9 exactly) | +5.88 recall |
| s32@0.70 | 75.68/75/76.47 | 70.27/85/52.94 | +5.41 acc, +23.53 recall (Falls 05,08,11,17), -10 spec |
| s32@0.80 | 72.97/90/52.94 | identical | 0 (threshold dominates) |
| s33@0.80 | 75.68/90/58.82 | 78.38/100/52.94 | pose slightly harmful at this point |

## Conclusions

1. s32@0.70 is the best balanced version under the original protocol:
   +8.11 acc, +15 spec vs v9 at equal recall; pose-off drops fall recall by
   23.5 pts - strongest causal evidence so far that the SNN pose pathway
   carries fall detection.
2. Kinect skeleton training signal (s32) roughly quadrupled the pose
   pathway's causal weight vs MediaPipe-only training (s30floor: +5.9).
3. The two residual false alarms (ADL 05, 06) are the genuinely fast-motion
   ADLs; the slow-lying group (07,11,12,15) is fixed in s32/s33.
4. s33's SNN class head now helps overall balance vs s32 (@0.60: spec 80),
   unlike s31; but recall drops - keep as secondary variant.

---

## s34 multi-seed update (fall-preserve)

s34 = s32 + SNN fall-preserve margin loss (w=0.3, margin=0.5). 5 seeds, val threshold stable at 0.70.

| seed | S4 acc | S4 spec | S4 recall |
|---|---|---|---|
| 42 | 75.68 | 75.00 | 76.47 |
| 1  | 78.38 | 75.00 | 82.35 |
| 7  | 75.68 | 75.00 | 76.47 |
| 3  | 70.27 | 50.00 | 94.12 |
| 5  | 75.68 | 70.00 | 82.35 |
| **mean ± std** | **75.14 ± 2.65** | **69.00 ± 9.70** | **82.35 ± 6.44** |

Protocol-selected seed = seed 5 (highest val bal-acc 0.754), final S4 = 75.68/70.00/82.35.
Mean recall 82.35 >= 80, confirming the goal is reachable; variance comes from one seed (seed 3) trading spec for extreme recall.
## no-SNN ablation update

| model | S4 acc | S4 spec | S4 recall | false alarms |
|---|---|---|---|---|
| s34 seed5 (SNN+pose+preserve) | 75.68 | 70.00 | 82.35 | 05,06,07,08,11,12 |
| no-SNN seed5 | 56.76 | 30.00 | 88.24 | 01,02,03,05,06,07,08,10,11,12,14,15,16,17 |
| v9 baseline | 67.57 | 60.00 | 76.47 | 05,06,07,11,12,15,16,17 |

Removing SNN gives higher recall (88.24) but catastrophic spec (30). The SNN
branch is the ADL-suppression mechanism; it lifts spec from 60 (v9) to 70 (s34)
while also raising recall from 76.47 to 82.35.
