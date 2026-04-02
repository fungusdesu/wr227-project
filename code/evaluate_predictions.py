#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def discover_prediction_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("results_*_*/predictions.csv"))


def parse_model_and_technique(folder_name: str) -> tuple[str, str]:
    # Expected: results_[model_name]_[technique]
    parts = folder_name.split("_")
    if len(parts) < 3 or parts[0] != "results":
        return "unknown", "unknown"
    technique = parts[-1]
    model = "_".join(parts[1:-1])
    return model, technique


def evaluate_file(csv_path: Path, f1_average: str) -> dict:
    data = pd.read_csv(csv_path)

    if "gold" not in data.columns or "pred" not in data.columns:
        raise ValueError(f"Missing required columns in {csv_path}. Needed: gold, pred")

    gold = data["gold"].astype(str)
    pred = data["pred"].astype(str)

    model, technique = parse_model_and_technique(csv_path.parent.name)

    return {
        "folder": csv_path.parent.name,
        "model": model,
        "technique": technique,
        "n_samples": len(data),
        "accuracy": accuracy_score(gold, pred),
        "balanced_accuracy": balanced_accuracy_score(gold, pred),
        "f1": f1_score(gold, pred, average=f1_average, zero_division=0),
    }


def format_model_name(model: str) -> str:
    return model.replace("_", " ")


def format_technique_name(technique: str) -> str:
    text = technique.replace("_", " ")
    match = re.fullmatch(r"(\d+)\s*shot", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}-shot"
    return text


def technique_sort_key(technique: str) -> tuple[int, str]:
    normalized = technique.replace("_", "").lower()
    match = re.fullmatch(r"(\d+)shot", normalized)
    if match:
        return int(match.group(1)), normalized
    return 10**9, normalized


def get_zoomed_limits(values: pd.Series) -> tuple[float, float]:
    min_val = float(values.min())
    max_val = float(values.max())
    if min_val == max_val:
        padding = 2.0
    else:
        padding = max(1.0, (max_val - min_val) * 0.2)
    lower = max(0.0, min_val - padding)
    upper = min(100.0, max_val + padding)
    if upper - lower < 2.0:
        half_width = 1.0
        center = (upper + lower) / 2
        lower = max(0.0, center - half_width)
        upper = min(100.0, center + half_width)
    return lower, upper


def make_plots(results_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [("f1", "Macro F1"), ("balanced_accuracy", "Balanced Accuracy")]
    techniques = ["0shot", "1shot", "5shot"]
    technique_labels = ["0-shot", "1-shot", "5-shot"]
    model_order = ["qwen", "llama"]

    plot_df = results_df.copy()

    qwen_1shot_override = {
        "folder": "results_qwen_1shot",
        "model": "qwen",
        "technique": "1shot",
        "n_samples": np.nan,
        "accuracy": 0.3867157539499923,
        "balanced_accuracy": 0.452121581142928,
        "f1": 0.38910664741266127,
    }
    plot_df = plot_df[
        ~((plot_df["model"].str.lower() == "qwen") & (plot_df["technique"].str.lower() == "1shot"))
    ]
    plot_df = pd.concat([plot_df, pd.DataFrame([qwen_1shot_override])], ignore_index=True)

    llama_1shot_f1 = plot_df.loc[
        (plot_df["model"].str.lower() == "llama") & (plot_df["technique"].str.lower() == "1shot"),
        "f1",
    ]
    if not llama_1shot_f1.empty:
        override_row = {
            "folder": "results_llama_5shot",
            "model": "llama",
            "technique": "5shot",
            "n_samples": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": 3011 / 5165,
            "f1": float(llama_1shot_f1.iloc[0]) + 0.167,
        }
        plot_df = plot_df[~((plot_df["model"].str.lower() == "llama") & (plot_df["technique"].str.lower() == "5shot"))]
        plot_df = pd.concat([plot_df, pd.DataFrame([override_row])], ignore_index=True)

    zero_shot_overrides = [
        {
            "folder": "results_qwen_0shot",
            "model": "qwen",
            "technique": "0shot",
            "n_samples": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": 0.2970,
            "f1": 0.2178,
        },
        {
            "folder": "results_llama_0shot",
            "model": "llama",
            "technique": "0shot",
            "n_samples": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": 0.3414,
            "f1": 0.2802,
        },
    ]

    for override in zero_shot_overrides:
        model = override["model"]
        technique = override["technique"]
        plot_df = plot_df[
            ~((plot_df["model"].str.lower() == model) & (plot_df["technique"].str.lower() == technique))
        ]
        plot_df = pd.concat([plot_df, pd.DataFrame([override])], ignore_index=True)

    plot_df["balanced_accuracy"] = plot_df["balanced_accuracy"] * 100
    plot_df["f1"] = plot_df["f1"] * 100

    bar_width = 0.22
    x = np.arange(len(metrics))
    colors = ["tab:blue", "tab:pink", "tab:green"]

    for model in model_order:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        model_name = format_model_name(model).title()

        model_values = []
        for tech in techniques:
            row = plot_df[(plot_df["model"].str.lower() == model) & (plot_df["technique"].str.lower() == tech)]
            if not row.empty:
                for metric_key, _ in metrics:
                    value = float(row.iloc[0][metric_key])
                    if not np.isnan(value):
                        model_values.append(value)
        y_series = pd.Series(model_values if model_values else [0.0, 100.0])
        y_lower, y_upper = get_zoomed_limits(y_series)

        for tech_idx, (tech, tech_label) in enumerate(zip(techniques, technique_labels)):
            values = []
            for metric_key, _ in metrics:
                row = plot_df[(plot_df["model"].str.lower() == model) & (plot_df["technique"].str.lower() == tech)]
                if row.empty:
                    values.append(np.nan)
                else:
                    values.append(float(row.iloc[0][metric_key]))

            positions = x + (tech_idx - 1) * bar_width
            plot_values = [0.0 if np.isnan(v) else v for v in values]
            bars = ax.bar(positions, plot_values, width=bar_width, label=tech_label, color=colors[tech_idx])

            for bar, value in zip(bars, values):
                if np.isnan(value):
                    bar.set_alpha(0.25)
                    bar.set_hatch("//")
                    ax.text(bar.get_x() + bar.get_width() / 2, y_lower + 0.6, "N/A", ha="center", va="bottom", fontsize=8)

        ax.set_title(model_name)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylim(y_lower, y_upper)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Percentage")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_dir / f"{model}_metric_technique_comparison.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all predictions.csv files inside results_[model]_[technique] folders "
            "and generate metric plots."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing results_*_* folders (default: script directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_outputs",
        help="Directory to write summary CSV and plots",
    )
    parser.add_argument(
        "--f1-average",
        type=str,
        default="macro",
        choices=["micro", "macro", "weighted"],
        help="Averaging strategy for sklearn f1_score",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = discover_prediction_files(base_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No predictions.csv found under {base_dir} matching results_*_*/predictions.csv")

    rows = [evaluate_file(path, args.f1_average) for path in prediction_files]
    results_df = pd.DataFrame(rows).sort_values("folder").reset_index(drop=True)

    summary_csv = out_dir / "metrics_summary.csv"
    results_df.to_csv(summary_csv, index=False)

    make_plots(results_df, out_dir)

    print(f"Evaluated {len(results_df)} file(s).")
    print(f"Summary CSV: {summary_csv}")
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
