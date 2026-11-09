# memristor-fall-detection

Code for the memristor-based fall-detection system described in the manuscript
(*Compositionally Complex Spinel-Ferrite Memristors …*). The detector combines a
CNN + GRU backbone with an SNN / LIF-inspired temporal branch, and maps the
network weights onto conductance levels measured on Pt/NiCuZnCoMnFe-O/Pt devices.

## Repository layout

```
automatic_fall_detector.py                     flattened entry point (train / infer / crossbar / SNN quantisation)
requirements.txt                               Python dependencies
VERSIONS.md                                    version registry: which directory produced which result

5.9new/crossbar+/CNN+SNN/                      original experiment tree
    README.md                                  design notes (Chinese)
    s30_pose/                                  pipeline used for the paper
        code/automatic_fall_detector.py        main pipeline
        scripts/queue_loso_v1000.py            queued LOSO runs, ideal vs non-ideal ablation
        scripts/queue_loso_dist_v1001.py       missing factorisation steps of the same ablation
        scripts/repro_v1001_protocol.py        reproduction of the locked main-table protocol
        v1000/                                 evaluation outputs (input_noise / no_snn / recall_priority)
        v1001/per_fold.csv                     main-table numbers (data only, no generator)
        v1002/                                 SNN branch: weight quantisation onto measured levels

tools/                                         one-off helpers (Word/PPT export, frame dumps)

dft/                                           Quantum ESPRESSO inputs, relaxed structures and
                                               plot-ready data behind Figure 3(a-e)
    README.md                                  model, settings, every number quoted in the text
```

The other `v*` directories are frozen snapshots kept for provenance; they are not
needed to reproduce the paper and are kept only so that the exact code behind each
reported number can be recovered.

## Environment

* Python 3.10
* PyTorch 2.5.1 + cu121, single GTX 1650 (8 GB)
* conda environment used during the experiments: `my_yizu_3.10`

```bash
pip install -r requirements.txt
```

## Data

Datasets and checkpoints are **not** included in the repository (they are large
and partly subject to third-party terms). To run the pipeline you need:

* GMDCSA-24 (public fall-detection dataset) for training and LOSO evaluation;
* the Zenodo fall videos used as the external validation set;
* the measured device data (HRS/LRS distributions, EPSC, LTP/LTD, PPF) for the
  non-ideal simulation;
* a trained checkpoint, or enough data to train one.

All of these are passed through command-line flags. Run

```bash
python automatic_fall_detector.py --help
```

for the full list; the important ones are `--video`, `--external-labeled-dir`,
`--external-val-labeled-dir`, `--data-dir`, `--model-path` and
`--snn-weight-levels`.

## Reproducing the paper

1. **LOSO non-ideality ablation** (ideal vs measured distributions, device
   dynamics, readout noise, SNN binary / multi-level weights):

   ```bash
   python "5.9new/crossbar+/CNN+SNN/s30_pose/scripts/queue_loso_v1000.py"
   python "5.9new/crossbar+/CNN+SNN/s30_pose/scripts/queue_loso_dist_v1001.py"
   ```

2. **Locked main-table protocol** (Zenodo validation + in-fold subject, F1-max
   threshold capped at 0.70):

   ```bash
   python "5.9new/crossbar+/CNN+SNN/s30_pose/scripts/repro_v1001_protocol.py"
   ```

3. **SNN weight mapping** onto measured conductance states:

   ```bash
   python "5.9new/crossbar+/CNN+SNN/s30_pose/v1002/code/automatic_fall_detector.py" \
       --mode train_infer --snn-weight-levels <levels> ...
   ```

`VERSIONS.md` records which snapshot produced each number, and
`s30_pose/v1001/per_fold.csv` holds the per-fold metrics of the main table.

Device data, trained checkpoints and the pose cache needed for exact reproduction live in `data/`, `models/` and `pose_cache/`; see [DATA.md](DATA.md) for provenance, read voltages and what is intentionally not included.

The density-functional calculations behind Figure 3(a-e) are shipped separately in `dft/` (inputs, relaxed structures, plot-ready data); settings and the mapping from each quoted number to its source file are in [dft/README.md](dft/README.md).

