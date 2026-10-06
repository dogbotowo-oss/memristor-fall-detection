from pathlib import Path
import runpy

source = Path(r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\noise_robustness_eval.py')
namespace = runpy.run_path(str(source), run_name='noise_sweep_v999_input')
namespace = namespace['main'].__globals__
namespace['V8_DIR'] = Path(r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v999')
namespace['CODE_PATH'] = namespace['V8_DIR'] / 'code' / 'automatic_fall_detector.py'
namespace['MODEL_PATH'] = namespace['V8_DIR'] / 'models' / 'automatic_fall_event_detector_c1_s34_data1_s11.pth'
namespace['DATA_DIR'] = Path(r'F:\12.18-2\data1')
namespace['THRESHOLD'] = 0.65
namespace['OUTPUT_DIRS'] = {
    'none': namespace['V8_DIR'] / 'reports' / 'noise_input_none' / 'shuju',
    'gaosi': namespace['V8_DIR'] / 'reports' / 'noise_input_gaussian' / 'shuju',
    'possion': namespace['V8_DIR'] / 'reports' / 'noise_input_poisson' / 'shuju',
    'salt': namespace['V8_DIR'] / 'reports' / 'noise_input_saltpepper' / 'shuju',
}
pose_cache = Path(r'F:\12.18-2\pose_cache')
videos = sorted(Path(r'F:\12.18-2\gmdcsa_subject4_test\Subject 4').rglob('*.mp4'))
missing = [video for video in videos if not (pose_cache / f'subject4_{video.parent.name.lower()}_{video.stem}.npz').exists()]
if missing:
    raise RuntimeError(f'Cannot reproduce pose-aware v999: missing {len(missing)} Subject4 pose caches in {pose_cache}. No evaluation started.')
raise RuntimeError('Evaluation adapter requires pose-aware inference before execution; legacy v8 inference omits pose_frame_features.')
