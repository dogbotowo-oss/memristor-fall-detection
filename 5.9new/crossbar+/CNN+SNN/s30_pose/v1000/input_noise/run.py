import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('F:/12.18-2')
S30 = ROOT / '5.9new/crossbar+/CNN+SNN/s30_pose'
OUT = Path(__file__).resolve().parent
CACHE = ROOT / 'pose_cache'
DATA = ROOT / 'data1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    subjects = [int(arg) for arg in sys.argv[1:]]
    torch.set_num_threads(2 if subjects else 4)
    evaluator = load('noise_evaluator', S30 / 'analysis/strict_eval_s30.py')
    evaluator.CODE_PATH = S30 / 'code/automatic_fall_detector.py'
    module = evaluator.load_module()
    evaluator.DATA_DIR = DATA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with (S30 / 'v1000/no_snn/comparison/v1000_loso_snn_vs_no_snn_fold_metrics.csv').open(encoding='utf-8-sig') as stream:
        folds = [row for row in csv.DictReader(stream) if row['scheme'] == 'balanced']
    if subjects:
        folds = [row for row in folds if int(row['test_subject']) in subjects]
    gmdcsa = next((ROOT / '测试集/13354453/ekramalam').glob('GMDCSA24*/ekramalam-GMDCSA24*5abac76'))
    jobs = []
    for row in folds:
        subject = int(row['test_subject'])
        snn = row['configuration'] == 'SNN'
        tag = f'c1_s34_data1_loso_ts{subject}_s11' if snn else f'c1_s34_data1_v1000_loso_ts{subject}_nosnn_s11'
        model_path = (S30 if snn else S30 / 'v1000/no_snn') / 'models' / f'automatic_fall_event_detector_{tag}.pth'
        assert model_path.is_file(), model_path
        video_root = ROOT / 'gmdcsa_subject4_test/Subject 4' if subject == 4 else gmdcsa / f'Subject {subject}'
        videos = sorted(module.list_video_files(video_root))
        assert len(videos) == int(row['n'])
        for video in videos:
            cache = module.find_pose_cache_file(video, CACHE)
            assert cache is not None, video
            with np.load(cache, allow_pickle=False) as saved:
                points = saved['keypoints']
                assert points.ndim == 3 and points.shape[1:] == (33, 3) and len(points) > 0
                assert np.isfinite(points).all(), cache
        jobs.append((row, model_path, videos))
    for pattern in ('HRS.csv', 'LRS.csv', 'EPSC.csv', 'LTP*.csv', 'LTD*.csv'):
        assert list(DATA.glob(pattern)), pattern
    audit_name = 'audit.json' if not subjects else f"audit_S{'_'.join(map(str, subjects))}.json"
    write_json(OUT / audit_name, {'status': 'inputs_verified', 'device': str(device), 'videos': 160, 'models': 8, 'pose_policy': 'fixed clean pose; noise on resized grayscale image before Crossbar', 'data_files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in DATA.glob('*.csv')}, 'model_files_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for _, path, _ in jobs}, 'folds': folds})
    print('AUDIT PASS: 8 models, 160 videos, 160 valid pose caches, data1 CSVs', flush=True)
    for row, model_path, videos in jobs:
        model, config = module.load_model_checkpoint(model_path, device)
        config['pose_cache_dir'] = str(CACHE)
        assert bool(config['use_snn_temporal_branch']) == (row['configuration'] == 'SNN')
        assert config['use_crossbar'] and config['use_device_dynamics']
        predictions, metrics = evaluator.evaluate_at_threshold(module, model, config, videos, float(row['threshold']), device)
        name = f"S{row['test_subject']}_{row['configuration']}"
        write_json(OUT / f'baseline_{name}.json', {'metrics': metrics, 'predictions': predictions})
        assert all(metrics[key] == int(row[key]) for key in ('tp', 'tn', 'fp', 'fn')), f'Baseline mismatch: {name}: {metrics}'
        print(f'BASELINE PASS {name}: {metrics}', flush=True)
        del model
    print('ALL BASELINES PASS; starting noise sweep', flush=True)
    levels = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    original_read = module.read_video_frames
    for row, model_path, videos in jobs:
        model, config = module.load_model_checkpoint(model_path, device)
        config['pose_cache_dir'] = str(CACHE)
        for noise in ('gaussian', 'saltpepper', 'poisson'):
            for level_index, level in enumerate(levels):
                for repeat in range(5 if level_index else 1):
                    filename = OUT / f"S{row['test_subject']}_{row['configuration']}_{noise}_{level_index}_{repeat}.json"
                    if filename.exists():
                        continue
                    def noisy_read(video_path, **kwargs):
                        frames, fps = original_read(video_path, **kwargs)
                        if not level_index:
                            return frames, fps
                        identity = f"{row['test_subject']}|{video_path.parent.name}|{video_path.stem}|{noise}|{level_index}|{repeat}"
                        rng = np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16))
                        frames = frames.astype(np.float32, copy=True)
                        if noise == 'gaussian':
                            frames += rng.normal(0, 0.30 * level, frames.shape).astype(np.float32)
                        elif noise == 'saltpepper':
                            mask = rng.random(frames.shape) < level
                            salt = rng.random(frames.shape) < 0.5
                            frames[mask & salt] = 1
                            frames[mask & ~salt] = 0
                        else:
                            peak = 220 * (6 / 220) ** (level_index / 7)
                            frames = rng.poisson(np.clip(frames, 0, 1) * peak).astype(np.float32) / peak
                        return np.clip(frames, 0, 1), fps
                    module.read_video_frames = noisy_read
                    try:
                        predictions, metrics = evaluator.evaluate_at_threshold(module, model, config, videos, float(row['threshold']), device)
                    finally:
                        module.read_video_frames = original_read
                    write_json(filename, {'subject': int(row['test_subject']), 'configuration': row['configuration'], 'noise': noise, 'level': level, 'repeat': repeat, 'metrics': metrics, 'predictions': predictions})
                    print(f'SAVED {filename.name}', flush=True)
        del model
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
