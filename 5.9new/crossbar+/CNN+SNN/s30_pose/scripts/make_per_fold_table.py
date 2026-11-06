"""Rebuild v1001/per_fold.csv from the per-fold report JSONs.

The v1001 directory is a frozen snapshot of the paper's main table.  Its
per-fold reports live in v1001/reports/*_test_metrics.json; this script only
flattens them into the single per_fold.csv used by the manuscript table, so
the table can be regenerated (and audited) without hand editing.

Usage:
    python make_per_fold_table.py [--s30 <path to s30_pose>] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FOLDER = {'snn': 'SNN', 'nosnn': 'no-SNN'}
COLUMNS = ['model', 'fold', 'val_threshold', 'val_fall_recall', 'val_adl_specificity',
           'val_F1', 'test_accuracy', 'test_balanced_accuracy', 'test_adl_specificity',
           'test_fall_recall', 'tp', 'tn', 'fp', 'fn']


def collect(reports: Path) -> list[list[str]]:
    rows = []
    for model in ('snn', 'nosnn'):
        for fold in (1, 2, 3, 4):
            path = reports / f'{model}_fold{fold}_test_metrics.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            r = json.loads(path.read_text(encoding='utf-8'))
            if r['fold'] != fold:
                raise ValueError(f'{path}: fold field {r["fold"]} != {fold}')
            rows.append([
                model, str(fold),
                f'{r["val_threshold"]:.2f}',
                f'{r["val_fall_recall"] * 100:.1f}',
                f'{r["val_adl_specificity"] * 100:.1f}',
                f'{r["val_f1"]:.4f}',
                f'{r["accuracy"]:.6f}',
                f'{r["balanced_accuracy"]:.6f}',
                f'{r["adl_specificity"]:.6f}',
                f'{r["fall_recall"]:.6f}',
                str(r['tp']), str(r['tn']), str(r['fp']), str(r['fn']),
            ])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--s30', default=str(Path(__file__).resolve().parents[1]),
                    help='path to s30_pose (default: parent of this script)')
    ap.add_argument('--check', action='store_true',
                    help='only verify the committed per_fold.csv')
    args = ap.parse_args()

    s30 = Path(args.s30)
    reports = s30 / 'v1001' / 'reports'
    out = s30 / 'v1001' / 'per_fold.csv'
    rows = collect(reports)
    text = ','.join(COLUMNS) + '\n' + '\n'.join(','.join(r) for r in rows) + '\n'

    if args.check:
        current = out.read_text(encoding='utf-8-sig')
        if current != text:
            print('MISMATCH: per_fold.csv does not match the report JSONs', file=sys.stderr)
            return 1
        print(f'OK: {out.name} reproduced from {len(rows)} report files')
        return 0

    out.write_text(text, encoding='utf-8', newline='\n')
    print(f'wrote {out} ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
