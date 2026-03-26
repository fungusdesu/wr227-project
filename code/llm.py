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

def sanitize_row(row_dict: dict) -> dict:
    row = dict(row_dict)
    row.pop("id_student", None)
    return row

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


def read_clean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    return df


def read_y_labels(y_path: Path) -> list[str]:
    y_df = read_clean_csv(y_path)

    if y_df.shape[1] == 0:
        raise ValueError("y_test.csv has no usable columns.")

    y_col = y_df.columns[0]
    y = y_df[y_col].astype(str).str.strip().tolist()

    normalized = []
    for value in y:
        label = normalize_label(value)
        if label is None:
            raise ValueError(
                f"Unexpected label in label CSV: {value!r}. "
                f"Allowed labels: {LABELS}"
            )
        normalized.append(label)

    return normalized


def load_few_shot_examples(
    shot_x_path: Path | None,
    shot_y_path: Path | None,
    shot_start_index: int,
    num_shots: int,
) -> list[tuple[dict, str]]:
    if num_shots == 0:
        return []

    if shot_x_path is None or shot_y_path is None:
        raise ValueError("num-shots > 0 requires both --shot-x and --shot-y.")

    x_shot_df = read_clean_csv(shot_x_path)
    y_shot = read_y_labels(shot_y_path)

    if len(x_shot_df) != len(y_shot):
        raise ValueError(
            f"Row count mismatch: shot X has {len(x_shot_df)} rows, "
            f"but shot y has {len(y_shot)} rows."
        )

    if shot_start_index < 0:
        raise ValueError("shot-start-index must be >= 0.")

    shot_end_index = shot_start_index + num_shots
    if shot_end_index > len(x_shot_df):
        raise IndexError(
            f"Requested shots [{shot_start_index}:{shot_end_index}] exceed "
            f"available shot rows ({len(x_shot_df)})."
        )

    examples = []
    for i in range(shot_start_index, shot_end_index):
        examples.append((x_shot_df.iloc[i].to_dict(), y_shot[i]))

    return examples

def load_balanced_few_shot_examples(
    shot_x_path: Path,
    shot_y_path: Path,
) -> list[tuple[dict, str]]:
    x_shot_df = read_clean_csv(shot_x_path)
    y_shot = read_y_labels(shot_y_path)

    if len(x_shot_df) != len(y_shot):
        raise ValueError(
            f"Row count mismatch: shot X has {len(x_shot_df)} rows, "
            f"but shot y has {len(y_shot)} rows."
        )

    examples = []
    used_indices = set()

    # one example per class
    for target_label in LABELS:
        for i, y in enumerate(y_shot):
            if i in used_indices:
                continue
            if y == target_label:
                examples.append((x_shot_df.iloc[i].to_dict(), y))
                used_indices.add(i)
                break

    # add one extra example from the first unused row
    for i, y in enumerate(y_shot):
        if i not in used_indices:
            examples.append((x_shot_df.iloc[i].to_dict(), y))
            break

    return examples

def build_prompt(
    base_prompt: str,
    row_dict: dict,
    few_shot_examples: list[tuple[dict, str]],
) -> str:
    current_row_json = json.dumps(row_dict, ensure_ascii=False)

    parts = [
        base_prompt,
        "Allowed labels: Distinction, Pass, Withdrawn, Fail.",
        "Infer the label from the pattern in the examples.",
        "Do not default to the most common example label.",
        "Output exactly one line in this format:",
        "LABEL=<one of Distinction, Pass, Withdrawn, Fail>",
        "Do not output anything else.",
    ]

    if few_shot_examples:
        parts.append("")
        parts.append("Examples:")

        for idx, (example_x, example_y) in enumerate(few_shot_examples, start=1):
            example_x_json = json.dumps(sanitize_row(example_x), ensure_ascii=False)
            parts.extend(
                [
                    f"Example {idx}:",
                    f"Student record:\n{example_x_json}",
                    f"LABEL={example_y}",
                    "",
                ]
            )

    parts.extend(
        [
            "Now classify this record:",
            f"Student record:\n{current_row_json}",
            "Answer:",
        ]
    )

    return "\n".join(parts)


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

    # Remove ANSI escape sequences
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)

    # Only parse the region after the final task prompt, not the earlier examples
    start_markers = [
        "Now classify this record:",
        "Student record:",
    ]

    start_idx = -1
    for marker in start_markers:
        idx = output.rfind(marker)
        if idx > start_idx:
            start_idx = idx

    if start_idx != -1:
        output = output[start_idx:]

    # Cut off trailing runtime logs
    stop_markers = [
        "llama_memory_breakdown_print:",
        "Exiting...",
        "[ Prompt:",
    ]
    stop_idx = len(output)
    for marker in stop_markers:
        idx = output.find(marker)
        if idx != -1 and idx < stop_idx:
            stop_idx = idx
    output = output[:stop_idx]

    # First, prefer a LABEL=... answer in the final region
    m = re.search(
        r"LABEL\s*=\s*(Distinction|Pass|Withdrawn|Fail)",
        output,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_label(m.group(1))

    # Fallback: accept a bare label in the final region
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


def save_predictions_csv(rows: list[dict], out_path: Path) -> None:
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

    parser.add_argument(
        "--shot-x",
        default=None,
        help="Path to few-shot feature CSV, e.g. X_train.csv",
    )
    parser.add_argument(
        "--shot-y",
        default=None,
        help="Path to few-shot label CSV, e.g. y_train.csv",
    )
    parser.add_argument(
        "--shot-start-index",
        type=int,
        default=0,
        help="Starting row index for few-shot examples",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=0,
        help="Number of few-shot examples to prepend (0 for zero-shot, 1 for one-shot, 5 for five-shot)",
    )

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
        help="Label for this run, e.g. qwen2.5-3b_one-shot-1ex",
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
    x_test_path = Path(args.x_test).expanduser().resolve()
    y_test_path = Path(args.y_test).expanduser().resolve()
    plot_out = Path(args.plot_out).expanduser().resolve()
    predictions_out = Path(args.predictions_out).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()

    shot_x_path = Path(args.shot_x).expanduser().resolve() if args.shot_x else None
    shot_y_path = Path(args.shot_y).expanduser().resolve() if args.shot_y else None

    if not llama_cli.exists():
        raise FileNotFoundError(f"llama-cli not found: {llama_cli}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not x_test_path.exists():
        raise FileNotFoundError(f"X_test.csv not found: {x_test_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"y_test.csv not found: {y_test_path}")
    if shot_x_path is not None and not shot_x_path.exists():
        raise FileNotFoundError(f"shot-x not found: {shot_x_path}")
    if shot_y_path is not None and not shot_y_path.exists():
        raise FileNotFoundError(f"shot-y not found: {shot_y_path}")

    x_df = read_clean_csv(x_test_path)
    y_true = read_y_labels(y_test_path)

    if len(x_df) != len(y_true):
        raise ValueError(
            f"Row count mismatch: X_test.csv has {len(x_df)} rows, "
            f"but y_test.csv has {len(y_true)} rows."
        )

    if args.num_shots == 5:
        few_shot_examples = load_balanced_few_shot_examples(
            shot_x_path=shot_x_path,
            shot_y_path=shot_y_path,
        )
    else:
        few_shot_examples = load_few_shot_examples(
            shot_x_path=shot_x_path,
            shot_y_path=shot_y_path,
            shot_start_index=args.shot_start_index,
            num_shots=args.num_shots,
        )
    
    if few_shot_examples:
        print(
            f"Using {len(few_shot_examples)} few-shot example(s) "
            f"starting from row {args.shot_start_index} of the shot dataset."
        )

    correct = 0
    invalid_predictions = 0
    total = len(x_df)

    y_pred_for_metrics: list[str] = []
    prediction_rows: list[dict] = []

    for i, (_, row) in enumerate(x_df.iterrows(), start=1):
        row_dict = sanitize_row(row.to_dict())
        prompt = build_prompt(
            base_prompt=args.prompt,
            row_dict=row_dict,
            few_shot_examples=few_shot_examples,
        )

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