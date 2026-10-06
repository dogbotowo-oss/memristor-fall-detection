from pathlib import Path


source = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\strict_eval_v8.py")
target = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\paper_no_snn_v8final\reports\strict_eval_no_snn_v8final.py")
text = source.read_text(encoding="utf-8")
text = text.replace(
    'MODEL_PATH = V8_DIR / "models" / "automatic_fall_event_detector_cnn_snn_v8_adl_gate_penalty.pth"',
    'MODEL_PATH = V8_DIR / "reports" / "paper_no_snn_v8final" / "models" / "automatic_fall_event_detector_no_snn_v8final.pth"',
)
text = text.replace(
    'REPORTS_DIR = V8_DIR / "reports"',
    'REPORTS_DIR = V8_DIR / "reports" / "paper_no_snn_v8final" / "reports"',
)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text, encoding="utf-8")
