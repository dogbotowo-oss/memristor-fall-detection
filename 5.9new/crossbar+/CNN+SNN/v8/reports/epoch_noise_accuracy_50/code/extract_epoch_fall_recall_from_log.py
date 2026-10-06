from __future__ import annotations

import csv
import re
from pathlib import Path


LOG_PATH = Path(
    r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\epoch_noise_accuracy_50\logs\train_epoch_noise_50.stdout.log"
)
OUTPUT_PATH = Path(
    r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\epoch_noise_accuracy_50\shuju\epoch_fall_recall_from_log.csv"
)


EPOCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|.*?val_acc=(?P<val_acc>\d+\.\d+)\s+"
    r"val_bal=(?P<val_bal>\d+\.\d+)\s+val_normal=(?P<val_normal>\d+\.\d+)\s+"
    r"val_fall=(?P<val_fall>\d+\.\d+)"
)


def main() -> None:
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: list[dict[str, float]] = []
    for line in text:
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        val_acc = float(match.group("val_acc")) * 100.0
        val_bal = float(match.group("val_bal")) * 100.0
        val_normal = float(match.group("val_normal")) * 100.0
        val_fall = float(match.group("val_fall")) * 100.0
        rows.append(
            {
                "Epoch": epoch,
                "Validation_Accuracy_percent": round(val_acc, 4),
                "Balanced_Accuracy_percent": round(val_bal, 4),
                "ADL_Specificity_percent": round(val_normal, 4),
                "Fall_Recall_percent": round(val_fall, 4),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Epoch",
                "Validation_Accuracy_percent",
                "Balanced_Accuracy_percent",
                "ADL_Specificity_percent",
                "Fall_Recall_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
