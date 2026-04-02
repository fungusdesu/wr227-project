#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_llm_label(model: str, technique: str) -> str:
    model_name = model.strip().upper()
    tech = technique.strip().replace("_", "-")
    return f"{model_name} ({tech})"


def load_tradml(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Model", "Balanced Accuracy", "Macro F1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "Label": df["Model"].astype(str),
            "Group": "Traditional ML",
            "F1": df["Macro F1"].astype(float),
            "Balanced Accuracy": df["Balanced Accuracy"].astype(float),
        }
    )
    return out


def load_llm(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"model", "technique", "f1", "balanced_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "Label": [format_llm_label(m, t) for m, t in zip(df["model"], df["technique"])],
            "Group": "LLM",
            "F1": df["f1"].astype(float),
            "Balanced Accuracy": df["balanced_accuracy"].astype(float),
        }
    )
    return out


def create_plot(combined: pd.DataFrame, output_path: Path) -> None:
    combined = combined.copy()
    combined["order"] = np.arange(len(combined))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)
    metrics = ["F1", "Balanced Accuracy"]

    group_colors = {
        "Traditional ML": "#4C78A8",
        "LLM": "#F58518",
    }
    bar_colors = [group_colors[group] for group in combined["Group"]]

    y = np.arange(len(combined))

    for ax, metric in zip(axes, metrics):
        ax.barh(y, combined[metric], color=bar_colors)
        ax.set_title(metric)
        ax.set_xlabel("Score")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_xlim(0, 1)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(combined["Label"])
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    split_idx = combined[combined["Group"] == "Traditional ML"].shape[0] - 0.5
    for ax in axes:
        ax.axhline(split_idx, color="black", linewidth=1.2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors["Traditional ML"], label="Traditional ML"),
        plt.Rectangle((0, 0), 1, 1, color=group_colors["LLM"], label="LLM"),
    ]
    fig.suptitle("Combined Performance: Traditional ML vs LLM", y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tradml_csv = base_dir / "model_performance_results_py.csv"
    llm_csv = base_dir / "evaluation_outputs_csv" / "llm_metrics_summary.csv"
    output_path = base_dir / "plots" / "combined_tradml_llm_performance.png"

    tradml_df = load_tradml(tradml_csv)
    llm_df = load_llm(llm_csv)

    combined = pd.concat([tradml_df, llm_df], ignore_index=True)
    create_plot(combined, output_path)

    print(f"Traditional ML CSV: {tradml_csv}")
    print(f"LLM CSV: {llm_csv}")
    print(f"Saved combined plot to: {output_path}")


if __name__ == "__main__":
    main()
