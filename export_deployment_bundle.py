"""Copy notebook outputs into a deployment-ready Streamlit artifact bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REQUIRED_FILES = ["best_model_final.keras", "best_tokenizer_final.pkl", "label_encoder.pkl"]
OPTIONAL_FILES = ["run_summary.json", "final_test_metrics.csv", "summary_table.csv"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="OUTPUT_DIR produced by the training notebook")
    parser.add_argument("--output-dir", default="artifacts", help="Destination artifact directory")
    parser.add_argument("--max-len", type=int, default=300)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Training output is incomplete. Missing: " + ", ".join(missing))

    for name in REQUIRED_FILES + OPTIONAL_FILES:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    summary = {}
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    config = {
        "model_file": "best_model_final.keras",
        "tokenizer_file": "best_tokenizer_final.pkl",
        "label_encoder_file": "label_encoder.pkl",
        "max_len": args.max_len,
        "text_column": summary.get("text_column", "text"),
        "label_column": summary.get("label_column", "text_category"),
        "model_name": summary.get("best_model_name", "Best deep-learning model"),
        "task_type": "single_label_priority",
        "priority_order": summary.get(
            "priority_order", ["Problem", "Suggestion", "Appreciation", "Neutral"]
        ),
    }
    (output_dir / "deployment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Deployment bundle created at: {output_dir}")


if __name__ == "__main__":
    main()
