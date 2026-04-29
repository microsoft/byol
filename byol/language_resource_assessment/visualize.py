# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Visualization for translation quality metrics.

Generates heatmaps showing:
- Metrics by domain and translator
- Macro average comparison
- Domain win counts
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from .config import (
    PLOT_METRICS,
    SORT_METRIC,
    TRANSLATOR_DISPLAY_NAMES,
    CORE_KEYS,
)
from .io import load_jsonl


def _get_display_name(translator: str) -> str:
    """Get human-friendly display name for a translator."""
    return TRANSLATOR_DISPLAY_NAMES.get(translator, translator)


def _process_results_for_metric(
    results: list[dict],
    translator_names: list[str],
    metric_name: str,
) -> dict[str, dict[str, list[float]]]:
    """
    Process results to get scores by domain and translator.
    
    Returns:
        Nested dict: domain -> translator -> list of scores
    """
    domain_translator_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    
    for result in results:
        domain = result["domain"]
        for translator in translator_names:
            if translator in result and result[translator].get(metric_name) is not None:
                score = result[translator].get(metric_name, 0)
                domain_translator_scores[domain][translator].append(score)
    
    return domain_translator_scores


def _get_global_translator_order(
    results: list[dict],
    translator_names: list[str],
    sort_metric: str,
) -> list[str]:
    """
    Determine translator order based on average score for a metric.
    
    Returns:
        List of translator names sorted by descending average score
    """
    domain_scores = _process_results_for_metric(results, translator_names, sort_metric)
    domains = sorted(set(r["domain"] for r in results))
    
    avg_scores = {}
    for translator in translator_names:
        all_scores = []
        for domain in domains:
            all_scores.extend(domain_scores[domain].get(translator, []))
        avg_scores[translator] = np.mean(all_scores) if all_scores else 0.0
    
    return sorted(translator_names, key=lambda t: avg_scores.get(t, 0.0), reverse=True)


def _create_heatmap_for_metric(
    results: list[dict],
    translator_names: list[str],
    metric_name: str,
    translator_order: list[str],
    output_dir: Path,
) -> None:
    """Create and save a heatmap for a specific metric."""
    domain_scores = _process_results_for_metric(results, translator_names, metric_name)
    domains = sorted(set(r["domain"] for r in results))
    
    # Calculate average scores for each translator
    translator_avg_scores = {}
    for translator in translator_names:
        all_scores = []
        for domain in domains:
            all_scores.extend(domain_scores[domain].get(translator, []))
        translator_avg_scores[translator] = np.mean(all_scores) if all_scores else 0.0
    
    # Get display names
    display_names = [_get_display_name(t) for t in translator_order]
    
    # Create score matrix
    score_matrix = []
    for domain in domains:
        row = []
        for translator in translator_order:
            scores = domain_scores[domain].get(translator, [])
            row.append(np.mean(scores) if scores else np.nan)
        score_matrix.append(row)
    
    domain_labels = [d.replace("_", " ") for d in domains]
    
    # Create dataframe
    score_df = pd.DataFrame(score_matrix, index=domain_labels, columns=display_names)
    
    # Macro average scores in order
    macro_avg_scores = [translator_avg_scores.get(t, np.nan) for t in translator_order]
    
    # Calculate domain wins
    num_domains = len(domains)
    translator_wins = defaultdict(int)
    
    if num_domains > 0 and translator_order and score_matrix:
        score_array = np.array(score_matrix)
        for domain_idx in range(num_domains):
            domain_row = score_array[domain_idx, :]
            if np.all(np.isnan(domain_row)):
                continue
            
            max_score = np.nanmax(domain_row)
            if np.isnan(max_score):
                continue
            
            winners = np.where(np.isclose(domain_row, max_score, equal_nan=False))[0]
            for idx in winners:
                translator_wins[translator_order[idx]] += 1
    
    wins_display = [
        f"{translator_wins[t]}/{num_domains}" if num_domains > 0 else "0/0"
        for t in translator_order
    ]
    
    # Clean metric name for display
    metric_display = metric_name.replace("tgt2src_", "").replace(
        "similarity_score", "Similarity Score"
    )
    
    # Create figure
    fig_height = 10 + 2  # Extra space for summary rows
    plt.figure(figsize=(14, fig_height))
    
    ax = sns.heatmap(
        score_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": f"Average {metric_display} (Higher is better)"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 14, "color": "black"},
    )
    
    # Set colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontsize(14)
    
    # Bold the highest value in each row
    for row_idx in range(len(score_matrix)):
        row_values = score_matrix[row_idx]
        if not np.all(np.isnan(row_values)):
            max_val = np.nanmax(row_values)
            if not np.isnan(max_val):
                max_indices = np.where(np.isclose(row_values, max_val, equal_nan=False))[0]
                for col_idx in max_indices:
                    text = ax.texts[row_idx * len(translator_order) + col_idx]
                    text.set_weight("bold")
                    text.set_fontsize(plt.rcParams["font.size"] * 1.2)
    
    # Adjust y-axis limits for summary rows
    ax.set_ylim(len(domains) + 2, 0)
    
    # Add Macro Average row
    y_macro = len(domains)
    rect_macro = Rectangle(
        (0, y_macro), len(translator_order), 1,
        facecolor="#d3d3d3", alpha=1.0, zorder=1,
    )
    ax.add_patch(rect_macro)
    
    for i, score_val in enumerate(macro_avg_scores):
        display_text = f"{score_val:.2f}" if not np.isnan(score_val) else "N/A"
        ax.text(
            i + 0.5, y_macro + 0.5, display_text,
            ha="center", va="center", weight="bold",
            fontsize=13, color="black", zorder=3,
        )
    
    # Add Domain Wins row
    y_wins = len(domains) + 1
    for i, win_text in enumerate(wins_display):
        ax.text(
            i + 0.5, y_wins + 0.5, win_text,
            ha="center", va="center", weight="normal",
            fontsize=plt.rcParams["font.size"] * 1.3, color="black", zorder=3,
        )
    
    # Update y-axis ticks
    y_ticks = [idx + 0.5 for idx in range(len(domains))] + [y_macro + 0.5, y_wins + 0.5]
    y_labels = domain_labels + ["Macro Average", "Domain Wins"]
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=14)
    
    # Style summary row labels
    yticklabels = ax.get_yticklabels()
    if len(yticklabels) > len(domains):
        yticklabels[len(domains)].set_weight("bold")
    
    # Add separator lines
    ax.axhline(len(domains), color="black", linewidth=2, zorder=2)
    ax.axhline(len(domains) + 1, color="black", linewidth=1.5, zorder=2)
    
    # Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    
    plt.xticks(rotation=10, ha="left", fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"ranking_{metric_name.replace('tgt2src_', '')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path}")


def generate_plots(
    input_file: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
    sort_metric: str = SORT_METRIC,
    excluded_translators: list[str] | None = None,
) -> None:
    """
    Generate heatmap visualizations for translation metrics.
    
    Args:
        input_file: Path to JSONL with computed metrics
        output_dir: Directory to save plots
        metrics: List of metrics to plot (default: PLOT_METRICS)
        sort_metric: Metric to use for global translator ordering
        excluded_translators: Translators to exclude from plots
    """
    if metrics is None:
        metrics = PLOT_METRICS
    
    if excluded_translators is None:
        excluded_translators = []
    
    # Load results
    results = load_jsonl(input_file)
    
    if not results:
        raise ValueError(f"No results found in {input_file}")
    
    # Get translator names (excluding specified ones)
    all_translators = [
        k for k in results[0]
        if k not in CORE_KEYS
    ]
    translator_names = sorted([
        t for t in all_translators
        if t not in excluded_translators
    ])
    
    print(f"Generating plots for {len(translator_names)} translators")
    print(f"Translators: {translator_names}")
    
    # Determine global translator order
    if translator_names:
        translator_order = _get_global_translator_order(results, translator_names, sort_metric)
        print(f"Translator order (by {sort_metric}): {translator_order}")
    else:
        translator_order = []
        print("Warning: No translators found")
        return
    
    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in metrics:
        print(f"\nGenerating plot for {metric}...")
        _create_heatmap_for_metric(
            results=results,
            translator_names=translator_names,
            metric_name=metric,
            translator_order=translator_order,
            output_dir=output_dir,
        )
    
    print(f"\nAll plots saved to {output_dir}")


__all__ = [
    "generate_plots",
]
