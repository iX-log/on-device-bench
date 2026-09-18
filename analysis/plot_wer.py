"""Plot quantization trade-offs by precision (RESULTS.md sessions 5 and 7).

Bar chart of aggregate word error rate for the whisper-base encoder at
each quantization precision, plus a combined disk-size-vs-WER trade-off
chart, both labeled with their values. Styled to match plot_sustained.py
so the repo's figures look consistent.
"""

import os

import matplotlib.pyplot as plt

CHARTS_DIR = "results/charts"

# RESULTS.md session 7, aggregate WER over 7 packed LibriSpeech windows.
WER_BY_PRECISION = {
    "fp16": 3.4,
    "int8": 3.8,
    "int4": 8.8,
}

# RESULTS.md session 5, .mlpackage disk size.
DISK_SIZE_MB_BY_PRECISION = {
    "fp16": 39.4,
    "int8": 19.8,
    "int4": 10.0,
}

PRECISION_COLORS = {
    "fp16": "steelblue",
    "int8": "tab:blue",
    "int4": "tab:purple",
}


def plot_wer_by_precision(wer_by_precision, out_path):
    precisions = list(wer_by_precision.keys())
    values = list(wer_by_precision.values())
    colors = [PRECISION_COLORS.get(p, "gray") for p in precisions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(precisions, values, color=colors, width=0.5)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(values) * 1.15)
    ax.set_xlabel("precision")
    ax.set_ylabel("aggregate WER (%)")
    ax.set_title("whisper-base encoder · aggregate WER by precision")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quantization_tradeoff(disk_size_mb_by_precision, wer_by_precision, out_path):
    precisions = list(disk_size_mb_by_precision.keys())
    sizes = [disk_size_mb_by_precision[p] for p in precisions]
    wers = [wer_by_precision[p] for p in precisions]
    colors = [PRECISION_COLORS.get(p, "gray") for p in precisions]

    fig, ax_size = plt.subplots(figsize=(10, 5))
    ax_wer = ax_size.twinx()

    bars = ax_size.bar(precisions, sizes, color=colors, width=0.5, alpha=0.6, label="disk size")
    for bar, value in zip(bars, sizes):
        ax_size.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.02,
            f"{value:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_size.set_ylim(0, max(sizes) * 1.2)
    ax_size.set_xlabel("precision")
    ax_size.set_ylabel("disk size (MB)")

    line, = ax_wer.plot(precisions, wers, color="black", marker="o", linewidth=1.5, label="aggregate WER")
    for x, value in zip(precisions, wers):
        ax_wer.text(x, value + max(wers) * 0.04, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax_wer.set_ylim(0, max(wers) * 1.3)
    ax_wer.set_ylabel("aggregate WER (%)")

    ax_size.set_title("whisper-base encoder: 4x smaller costs 2.6x the error rate (fp16 → int4)")
    ax_size.legend(handles=[bars, line], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    out_path = os.path.join(CHARTS_DIR, "wer-by-precision.png")
    plot_wer_by_precision(WER_BY_PRECISION, out_path)
    print(f"wrote {out_path}")

    out_path = os.path.join(CHARTS_DIR, "quantization-tradeoff.png")
    plot_quantization_tradeoff(DISK_SIZE_MB_BY_PRECISION, WER_BY_PRECISION, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
