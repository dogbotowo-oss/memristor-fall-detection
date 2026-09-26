import csv
import math
import statistics
from pathlib import Path


root = Path(__file__).resolve().parent
with (root / 'v1000_loso_snn_vs_no_snn_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
    summary = {row['configuration']: row for row in csv.DictReader(stream) if row['scheme'] == 'balanced'}
with (root / 'v1000_loso_snn_vs_no_snn_fold_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
    folds = [row for row in csv.DictReader(stream) if row['scheme'] == 'balanced']

metrics = [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced Accuracy'), ('adl_specificity', 'ADL specificity'), ('fall_recall', 'Fall recall')]
comparison = []
improvement = []
for metric, label in metrics:
    values = {}
    for configuration in ('no-SNN', 'SNN'):
        selected = [row for row in folds if row['configuration'] == configuration]
        assert len(selected) == 4 and sum(int(row['n']) for row in selected) == 160
        samples = []
        for row in selected:
            tn, fp, fn, tp = (int(row[key]) for key in ('tn', 'fp', 'fn', 'tp'))
            expected = {'accuracy': 100 * (tn + tp) / (tn + fp + fn + tp), 'balanced_accuracy': 50 * (tn / (tn + fp) + tp / (tp + fn)), 'adl_specificity': 100 * tn / (tn + fp), 'fall_recall': 100 * tp / (tp + fn)}[metric]
            assert math.isclose(expected, float(row[metric]), abs_tol=1e-9)
            samples.append(expected)
        mean, sd = statistics.mean(samples), statistics.stdev(samples)
        assert math.isclose(mean, float(summary[configuration]['macro_' + metric + '_mean']), abs_tol=1e-9)
        assert math.isclose(sd, float(summary[configuration]['macro_' + metric + '_sd']), abs_tol=1e-9)
        values[configuration] = (mean, sd)
    comparison.append([label, *[f'{value:.10f}' for value in (values['no-SNN'][0], values['SNN'][0], values['no-SNN'][1], values['SNN'][1])]])
    improvement.append([label, f"{values['SNN'][0] - values['no-SNN'][0]:.10f}"])

for filename, header, rows in [
    ('v1000_bar_comparison_macro.csv', ['Metric', 'no_SNN_mean_percent', 'SNN_mean_percent', 'no_SNN_sd_percentage_points', 'SNN_sd_percentage_points'], comparison),
    ('v1000_bar_improvement_macro.csv', ['Metric', 'SNN_minus_no_SNN_percentage_points'], improvement),
]:
    with (root / filename).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    with (root / filename).open(encoding='utf-8-sig', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 4
    print(root / filename)
