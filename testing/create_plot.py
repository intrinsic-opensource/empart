# Copyright 2025 Intrinsic Innovation LLC

"""
create_plot.py
================
This script generates comparison plots from performance data stored in a JSON file.
It visualizes metrics such as the number of parts and real-time factor for two methods:
- VHACD / CoACD
- Our method
"""

import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


###############################################################################
# 1.  Helper to pull one metric from either slice                             #
###############################################################################
def metric(series, key, factor=1.0):
    """Return list of metric values from a slice, applying scale if needed."""
    return [rec[key] * factor for rec in series]


###############################################################################
# 2.  Main plotting function                                                  #
###############################################################################
plt.rcParams.update(
    {
        "font.size": 16,  # base font size
        "axes.titlesize": 18,  # title of subplots
        "axes.labelsize": 16,  # x/y axis labels
        "xtick.labelsize": 14,  # x-axis tick labels
        "ytick.labelsize": 14,  # y-axis tick labels
        "legend.fontsize": 14,  # legend text
        "figure.titlesize": 20,  # overall figure title (if used)
        "axes.edgecolor": "black",
    }
)


def plot_all_metrics(vhacd, ours, output_basename, separate=False):
    """
    Generates and saves comparison plots from the provided data.

    Args:
        vhacd (list): A list of dictionary records for the VHACD method.
        ours (list): A list of dictionary records for your method.
        output_basename (str): The base filename for the saved plots (without extension).
        separate (bool): If True, creates separate subplots for each metric.

    Returns:
        None: Saves the plots as PNG and PDF files.
    """
    # X-axis: average Hausdorff error in millimetres
    x_vhacd = metric(vhacd, "err", factor=1_000)
    x_ours = metric(ours, "method_err", factor=1_000)

    # Define Y-axis metrics and their display names
    metrics = [
        (metric(ours, "num_hulls"), metric(vhacd, "num_hulls"), "Number of Parts"),
        (metric(ours, "rt_factor"), metric(vhacd, "rt_factor"), "Real‑Time Factor"),
    ]
    titles = ["Part Decomposition", "Simulation Performance"]

    # Labels keys for each subplot (first: num_hulls, second: rt_factor)
    label_keys = ["num_hulls", "rt_factor"]
    labels_ours_list = [[round(rec[key], 3) for rec in ours] for key in label_keys]
    labels_vhacd_list = [[round(rec[key], 3) for rec in vhacd] for key in label_keys]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4), dpi=500)
    for idx, (ax, (y_ours, y_vhacd, ylabel)) in enumerate(zip(axes, metrics)):
        # Scatter points
        ax.scatter(x_ours, y_ours, label="Our Method", s=80)
        ax.scatter(x_vhacd, y_vhacd, label="VHACD", s=80, marker="X")

        if idx == 1:
            # For second plot, place labels top-left
            ours_offset = (5, -5)
            vhacd_offset = (-35, 3)
            ha_ours, ha_vhacd, va = "left", "left", "top"
        else:
            ours_offset = (5, 5)
            vhacd_offset = (-30, -15)
            ha_ours, ha_vhacd, va = "right", "left", "bottom"

        # Annotate each point with the appropriate label and alignment
        labels_ours = labels_ours_list[idx]
        labels_vhacd = labels_vhacd_list[idx]
        for x, y, lbl in zip(x_ours, y_ours, labels_ours):
            ax.annotate(
                str(lbl),
                (x, y),
                textcoords="offset points",
                xytext=ours_offset,
                ha=ha_ours,
                va=va,
                fontsize=10,
            )
        for x, y, lbl in zip(x_vhacd, y_vhacd, labels_vhacd):
            ax.annotate(
                str(lbl),
                (x, y),
                textcoords="offset points",
                xytext=vhacd_offset,
                ha=ha_vhacd,
                va=va,
                fontsize=10,
            )
        # Trend lines
        sns.lineplot(x=x_ours, y=y_ours, ax=ax)
        sns.lineplot(x=x_vhacd, y=y_vhacd, ax=ax, linestyle="--")

        # Formatting per subplot
        if idx == 0:
            ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-", linewidth=0.7)
        ax.set_title(titles[idx])
        ax.set_xlabel("Avg Error Across Regions (mm)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=12)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.06 * (xmax - xmin), xmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    plt.rcParams["pdf.fonttype"] = 42  # For editable text in PDF
    plt.rcParams["ps.fonttype"] = 42
    plt.tight_layout()

    # Save figures using the output_basename
    png_file = f"{output_basename}.png"
    pdf_file = f"{output_basename}.pdf"
    plt.savefig(png_file, dpi=500, bbox_inches="tight")
    plt.savefig(pdf_file, dpi=500, bbox_inches="tight")
    print(f"✅ Plots saved to '{png_file}' and '{pdf_file}'")


def main():
    """Main function to parse arguments and run the plotting script."""
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from method performance data in a JSON file."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the input JSON file containing the performance records.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="comparison_plots",
        help="Basename for the output plot files (e.g., 'my_analysis'). Default is 'comparison_plots'.",
    )
    args = parser.parse_args()

    # 2. Load data from the specified JSON file
    try:
        with open(args.json_file, "r") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file '{args.json_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(
            f"❌ Error: Could not decode JSON from the file '{args.json_file}'. Please check its format."
        )
        return

    # 3. Split records by method
    #    Assumes the first 3 records are for VHACD and the rest are for our method.
    if len(records) < 4:
        print(
            "❌ Error: JSON file must contain at least 4 records (3 for VHACD, >=1 for ours)."
        )
        return

    vhacd = records[1:4]
    ours = records[4:]

    # 4. Generate the plots
    plot_all_metrics(vhacd, ours, args.output, separate=True)


if __name__ == "__main__":
    main()
