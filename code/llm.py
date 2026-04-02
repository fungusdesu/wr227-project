import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, recall_score


LABELS = ["Distinction", "Pass", "Withdrawn", "Fail"]


def normalize_label(text: str) -> str | None:
    if text is None:
        return None

    mapping = {
        "distinction": "Distinction",
        "pass": "Pass",
        "withdrawn": "Withdrawn",
        "fail": "Fail",
    }
    return mapping.get(text.strip().lower())


def read_y_labels(y_path: Path) -> list[str]:
    y_df = pd.read_csv(y_path)
    y_df = y_df.loc[:, ~y_df.columns.str.contains(r"^Unnamed", case=False, regex=True)]

    if y_df.shape[1] == 0:
        raise ValueError("y_test.csv has no usable columns.")

    y_col = y_df.columns[0]
    y = y_df[y_col].astype(str).str.strip().tolist()

    normalized = []
    for value in y:
        label = normalize_label(value)
        if label is None:
            raise ValueError(
                f"Unexpected label in y_test.csv: {value!r}. "
                f"Allowed labels: {LABELS}"
            )
        normalized.append(label)

    return normalized


def build_prompt(base_prompt: str, row_dict: dict) -> str:
    row_json = json.dumps(row_dict, ensure_ascii=False)

    return (
        f"{base_prompt}\n"
        f"Allowed labels: Distinction, Pass, Withdrawn, Fail.\n"
        f"Output exactly one line in this format:\n"
        f"LABEL=<one of Distinction, Pass, Withdrawn, Fail>\n"
        f"Do not output anything else.\n"
        f"Student record:\n{row_json}"
    )


def run_model(
    llama_cli: Path,
    model_path: Path,
    prompt: str,
    ngl: int,
    n_predict: int,
    ctx_size: int,
    timeout: int,
) -> str:
    cmd = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-st",
        "--simple-io",
        "--no-display-prompt",
        "--no-show-timings",
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx_size),
        "--temp",
        "0",
        "--top-k",
        "1",
        "-fa",
        "auto",
        "-ngl",
        str(ngl),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed with exit code {result.returncode}\n"
            f"OUTPUT:\n{result.stdout}"
        )

    return result.stdout or ""


def extract_prediction(output: str) -> str | None:
    if not output:
        return None

    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)

    m = re.search(
        r"LABEL\s*=\s*(Distinction|Pass|Withdrawn|Fail)",
        output,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_label(m.group(1))

    marker = "Student record:"
    idx = output.rfind(marker)
    if idx != -1:
        output = output[idx + len(marker):]

    for stop in [
        "llama_memory_breakdown_print:",
        "Exiting...",
        "[ Prompt:",
    ]:
        stop_idx = output.find(stop)
        if stop_idx != -1:
            output = output[:stop_idx]

    m = re.search(
        r"(Distinction|Pass|Withdrawn|Fail)",
        output,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_label(m.group(1))

    return None


def save_metrics_plot(macro_f1: float, balanced_acc: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Macro-F1", "Balanced Accuracy"]
    values = [macro_f1, balanced_acc]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_predictions_csv(
    rows: list[dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def append_summary_csv(
    out_path: Path,
    run_label: str,
    correct: int,
    total: int,
    accuracy: float,
    macro_f1: float,
    balanced_acc: float,
    invalid_predictions: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame(
        [
            {
                "run_label": run_label,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_acc,
                "invalid_predictions": invalid_predictions,
            }
        ]
    )

    if out_path.exists():
        row.to_csv(out_path, mode="a", header=False, index=False)
    else:
        row.to_csv(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", required=True, help="Path to llama-cli binary")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--x-test", required=True, help="Path to X_test.csv")
    parser.add_argument("--y-test", required=True, help="Path to y_test.csv")
    parser.add_argument("--prompt", required=True, help="Base prompt")
    parser.add_argument("--ngl", type=int, default=80, help="Number of GPU layers")
    parser.add_argument("--n-predict", type=int, default=32, help="Max generated tokens")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per row in seconds")
    parser.add_argument(
        "--plot-out",
        default="plots/metrics.png",
        help="Output PNG for the Macro-F1 / Balanced Accuracy plot",
    )
    parser.add_argument(
        "--predictions-out",
        default="results/predictions.csv",
        help="Output CSV for per-row predictions",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/summary.csv",
        help="CSV file to append one summary row per run",
    )
    parser.add_argument(
        "--run-label",
        default="run",
        help="Label for this run, e.g. qwen2.5-3b_zero-shot",
    )
    parser.add_argument(
        "--debug-first-n",
        type=int,
        default=3,
        help="Print raw output for the first N rows",
    )

    args = parser.parse_args()

    llama_cli = Path(args.llama_cli).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    x_path = Path(args.x_test).expanduser().resolve()
    y_path = Path(args.y_test).expanduser().resolve()
    plot_out = Path(args.plot_out).expanduser().resolve()
    predictions_out = Path(args.predictions_out).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()

    if not llama_cli.exists():
        raise FileNotFoundError(f"llama-cli not found: {llama_cli}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not x_path.exists():
        raise FileNotFoundError(f"X_test.csv not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_test.csv not found: {y_path}")

    x_df = pd.read_csv(x_path)
    x_df = x_df.loc[:, ~x_df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    y_true = read_y_labels(y_path)

    if len(x_df) != len(y_true):
        raise ValueError(
            f"Row count mismatch: X_test.csv has {len(x_df)} rows, "
            f"but y_test.csv has {len(y_true)} rows."
        )

    correct = 0
    invalid_predictions = 0
    total = len(x_df)

    y_pred_for_metrics: list[str] = []
    prediction_rows: list[dict] = []

    for i, (_, row) in enumerate(x_df.iterrows(), start=1):
        row_dict = row.to_dict()
        prompt = build_prompt(args.prompt, row_dict)

        raw_output = run_model(
            llama_cli=llama_cli,
            model_path=model_path,
            prompt=prompt,
            ngl=args.ngl,
            n_predict=args.n_predict,
            ctx_size=args.ctx_size,
            timeout=args.timeout,
        )

        pred = extract_prediction(raw_output)
        gold = y_true[i - 1]

        if i <= args.debug_first_n:
            print("\n===== RAW OUTPUT =====")
            print(repr(raw_output))
            print("======================\n")

        if pred == gold:
            correct += 1

        if pred is None:
            invalid_predictions += 1
            pred_for_metrics = "__INVALID__"
        else:
            pred_for_metrics = pred

        y_pred_for_metrics.append(pred_for_metrics)

        prediction_rows.append(
            {
                "row_index": i - 1,
                "gold": gold,
                "pred": pred,
                "raw_pred_for_metrics": pred_for_metrics,
                "is_correct": pred == gold,
            }
        )

        print(f"[{i}/{total}] pred={pred!r} gold={gold!r} correct={correct}")

    accuracy = correct / total if total > 0 else 0.0

    macro_f1 = f1_score(
        y_true,
        y_pred_for_metrics,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    balanced_acc = recall_score(
        y_true,
        y_pred_for_metrics,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print(f"\nCorrect predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Invalid predictions: {invalid_predictions}/{total}")

    save_predictions_csv(prediction_rows, predictions_out)
    save_metrics_plot(macro_f1, balanced_acc, plot_out)
    append_summary_csv(
        summary_csv,
        args.run_label,
        correct,
        total,
        accuracy,
        macro_f1,
        balanced_acc,
        invalid_predictions,
    )

    print(f"Saved predictions to: {predictions_out}")
    print(f"Saved metrics plot to: {plot_out}")
    print(f"Appended summary row to: {summary_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())