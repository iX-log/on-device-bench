"""Plot sustained-run latency over time with thermal state transitions.

Reads every results/device-pull/sustained-*.json and, for each, saves a
scatter of latency_ms vs elapsed_s with a rolling median overlay and
vertical lines marking thermal state transitions. Also saves two combined
charts: an overlay of the on-power and battery runs (RESULTS.md sessions 3
and 4) so the difference in thermal onset timing is visible on one axis,
and a cross-device chart normalising every run's rolling median to its own
first-minute baseline (percent slower), so devices with different absolute
latencies (RESULTS.md sessions 3/4/10) are directly comparable.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

DEVICE_PULL_DIR = "results/device-pull"
CHARTS_DIR = "results/charts"
ROLLING_WINDOW = 101  # samples, centered

# RESULTS.md sessions 3 (on charger), 4 (on battery), and 10 (external,
# iPhone 17) identify these runs by filename. Note: the 'notes' field
# inside the session 3/4 JSON files says "off power" for both, which
# contradicts RESULTS.md -- the labels below follow RESULTS.md as the
# source of truth; flag this if it matters. All four predate
# schema_version and its structured conditions (see format_conditions
# below), so this is exactly the kind of asserted-not-read mislabeling
# that schema 2 exists to catch.
RUN_LABELS = {
    "sustained-1788785293": "on-power",
    "sustained-1788866056": "battery",
    "sustained-fp16-1790151796": "cooled start",
    "sustained-fp16-1790102418": "warm start",
}

# Every schema-1 file's `device` field is a hardcoded marketing string, not
# a read value (see format_conditions). The session 3/4 files happen to be
# correct by coincidence; the session 10 files are not -- both say "iPhone
# 14 Pro Max" despite being an iPhone 17. This is the corrected label used
# for chart titles and legends; it does not touch the JSON.
DEVICE_NAMES = {
    "sustained-1788785293": "iPhone 14 Pro Max (A16, 6GB)",
    "sustained-1788866056": "iPhone 14 Pro Max (A16, 6GB)",
    "sustained-fp16-1790151796": "iPhone 17 (A19)",
    "sustained-fp16-1790102418": "iPhone 17 (A19)",
}

# plot_comparison() is specifically the session 3 vs 4 (on-power vs
# battery, same device) thermal-onset comparison. Scoped explicitly so
# that adding more sustained-*.json files later (e.g. session 10) doesn't
# silently pull them into a chart whose title and two-color legend assume
# exactly these two runs.
THERMAL_COMPARISON_STEMS = ("sustained-1788785293", "sustained-1788866056")

THERMAL_COLORS = {
    "nominal": "tab:green",
    "fair": "tab:orange",
    "serious": "tab:red",
    "critical": "darkred",
}


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["samples"])
    df["latency_roll"] = df["latency_ms"].rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
    return d, df


def format_conditions(meta):
    """Files with no `schema_version` key predate it (schema 1): `device`
    was a hardcoded marketing name and `notes` was free text asserting
    conditions rather than reading them. Schema 2 replaces `notes` with
    structured fields captured at run start -- this reads either shape."""
    version = meta.get("schema_version", 1)
    if version < 2:
        return meta.get("notes", "(no notes recorded)")

    level = meta.get("battery_level")
    level_str = f"{level:.0%}" if isinstance(level, (int, float)) and level >= 0 else "unknown"
    return (
        f"device: {meta.get('device')}; "
        f"low power mode: {meta.get('low_power_mode_enabled')}; "
        f"battery: {meta.get('battery_state')} ({level_str}); "
        f"thermal @ start: {meta.get('thermal_state_at_start')}; "
        f"network available @ start: {meta.get('network_available_at_start')} "
        f"(proxy for airplane mode -- iOS doesn't expose that state directly)"
    )


def thermal_transitions(df):
    """(state, elapsed_s) for each state change, skipping the initial state."""
    transitions = []
    prev = None
    for state, t in zip(df["thermal"], df["elapsed_s"]):
        if state != prev:
            if prev is not None:
                transitions.append((state, t))
            prev = state
    return transitions


def plot_single(path, meta, df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["elapsed_s"], df["latency_ms"], s=4, alpha=0.12, color="steelblue", linewidths=0)
    ax.plot(df["elapsed_s"], df["latency_roll"], color="black", linewidth=1.5, label=f"rolling median ({ROLLING_WINDOW})")

    ymax = df["latency_ms"].quantile(0.995)
    ax.set_ylim(0, ymax * 1.15)
    label_transform = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for state, t in thermal_transitions(df):
        color = THERMAL_COLORS.get(state, "gray")
        ax.axvline(t, color=color, linestyle="--", linewidth=1)
        ax.text(t, 0.97, f" {state} @ {t:.0f}s", color=color, va="top", ha="left", fontsize=8,
                 rotation=90, transform=label_transform)

    ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("latency (ms)")
    stem = os.path.splitext(os.path.basename(path))[0]
    label = RUN_LABELS.get(stem, "")
    device = DEVICE_NAMES.get(stem, meta["device"])
    title = f"{device} · {meta['model']} · {meta['precision']}"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(runs, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"on-power": "tab:blue", "battery": "tab:purple"}
    label_transform = ax.get_xaxis_transform()  # x in data coords, y in axes fraction

    for row, (path, meta, df) in enumerate(runs):
        stem = os.path.splitext(os.path.basename(path))[0]
        label = RUN_LABELS.get(stem, stem)
        color = colors.get(label, None)
        y_frac = 0.97 - 0.4 * row  # stagger rows so close transitions don't overlap
        ax.plot(df["elapsed_s"], df["latency_roll"], color=color, linewidth=1.5, label=f"{label} (rolling median)")
        for state, t in thermal_transitions(df):
            ax.axvline(t, color=color, linestyle="--", linewidth=1, alpha=0.7)
            ax.text(t, y_frac, f" {state} @ {t:.0f}s", color=color, va="top", ha="left", fontsize=7,
                     rotation=90, transform=label_transform)

    ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("latency (ms), rolling median")
    ax.set_title("Sustained run: thermal onset, on-power vs battery")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def normalized_series(df):
    """Rolling median as percent slower than this run's own first-minute
    median. Returns (pct_series, final_drift_pct), where final_drift_pct
    compares the last-minute median to the first-minute median -- the
    same first/last-minute convention RESULTS.md uses everywhere else."""
    baseline = df.loc[df["elapsed_s"] < 60, "latency_ms"].median()
    pct = (df["latency_roll"] - baseline) / baseline * 100
    end = df["elapsed_s"].max()
    last_minute = df.loc[df["elapsed_s"] >= end - 60, "latency_ms"].median()
    final_drift = (last_minute - baseline) / baseline * 100
    return pct, final_drift


def plot_normalized_comparison(runs, out_path):
    """Every run's rolling median normalised to its own first-minute
    baseline, on one axis, so devices with different absolute latencies
    (an A16 at ~42ms, an A19 at ~26ms) are directly comparable by shape:
    the A16's step vs the A19's creep. See DEVICE_NAMES for why device
    names come from that table rather than the (often mislabeled)
    `device` field in the raw JSON."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for path, meta, df in runs:
        stem = os.path.splitext(os.path.basename(path))[0]
        device = DEVICE_NAMES.get(stem, meta.get("device", stem))
        tag = RUN_LABELS.get(stem)
        name = f"{device}, {tag}" if tag else device
        pct, final_drift = normalized_series(df)
        ax.plot(df["elapsed_s"], pct, linewidth=1.5, label=f"{name}: {final_drift:+.1f}% final")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("latency, % slower than own first-minute median")
    ax.set_title("Sustained-run degradation, normalised across devices")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DEVICE_PULL_DIR, "sustained-*.json")))
    if not paths:
        print(f"no sustained-*.json files found in {DEVICE_PULL_DIR}")
        return

    runs = []
    for path in paths:
        meta, df = load_run(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(CHARTS_DIR, f"{stem}.png")
        plot_single(path, meta, df, out_path)
        print(f"wrote {out_path}")
        print(f"  conditions: {format_conditions(meta)}")
        runs.append((path, meta, df))

    thermal_runs = [r for r in runs if os.path.splitext(os.path.basename(r[0]))[0] in THERMAL_COMPARISON_STEMS]
    if len(thermal_runs) >= 2:
        out_path = os.path.join(CHARTS_DIR, "sustained_comparison.png")
        plot_comparison(thermal_runs, out_path)
        print(f"wrote {out_path}")

    if len(runs) >= 2:
        out_path = os.path.join(CHARTS_DIR, "cross-device-normalised.png")
        plot_normalized_comparison(runs, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
