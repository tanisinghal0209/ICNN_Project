import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "experiments" / "summary.json"
CSV_PATH = ROOT / "experiments" / "summary.csv"

with open(SUMMARY_PATH, "r", encoding="utf-8") as fh:
    data = json.load(fh)

fieldnames = ["experiment", "final_loss", "runtime_seconds", "model_type", "hidden_layers", "hidden_width", "learning_rate", "activation", "sample_count", "dimension", "distribution"]

rows = []
for item in data:
    row = {
        "experiment": item.get("experiment"),
        "final_loss": item.get("final_loss") if "final_loss" in item else item.get("final_f_loss"),
        "runtime_seconds": item.get("runtime_sec") if "runtime_sec" in item else item.get("runtime_seconds"),
        "model_type": item.get("model_type"),
        "hidden_layers": item.get("hidden_layers"),
        "hidden_width": item.get("hidden_width"),
        "learning_rate": item.get("learning_rate"),
        "activation": item.get("activation"),
        "sample_count": item.get("sample_count"),
        "dimension": item.get("dimension"),
        "distribution": item.get("distribution"),
    }
    rows.append(row)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {CSV_PATH}")
