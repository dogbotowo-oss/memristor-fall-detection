import csv
import json
import re
from pathlib import Path

OUT = Path.home() / 'Desktop/data/noise-duo'
SRC = Path(__file__).resolve().parent

METRIC_KEYS = ['threshold', 'total', 'accuracy', 'balanced_accuracy', 'adl_specificity',
               'fall_recall', 'precision', 'tp', 'tn', 'fp', 'fn',
               'false_alarm_count', 'missed_fall_count']
COLUMNS = ['subject', 'configuration', 'noise', 'level', 'repeat'] + METRIC_KEYS


def rows_from_json(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    metrics = data.get('metrics', {})
    if path.name.startswith('baseline_'):
        m = re.match(r'baseline_S(\d)_(SNN|no-SNN)\.json', path.name)
        subject, config = int(m.group(1)), m.group(2)
        noise, level, repeat = 'baseline', 0, 0
    else:
        subject = data['subject']
        config = data['configuration']
        noise = data['noise']
        level = data['level']
        repeat = data['repeat']
    row = {'subject': subject, 'configuration': config, 'noise': noise,
           'level': level, 'repeat': repeat}
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, '')
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(SRC.glob('*.json')):
        if path.name.startswith('audit'):
            continue
        rows.append(rows_from_json(path))
    rows.sort(key=lambda r: (r['subject'], r['configuration'], r['noise'], r['level'], r['repeat']))
    with (OUT / 'all.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for subject in (1, 2, 3, 4):
        subject_rows = [r for r in rows if r['subject'] == subject]
        with (OUT / f'S{subject}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(subject_rows)
        print(f'S{subject}.csv: {len(subject_rows)} rows')
    print(f'all.csv: {len(rows)} rows -> {OUT}')


if __name__ == '__main__':
    main()
