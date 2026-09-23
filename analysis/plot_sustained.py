"""Plot sustained-run latency over time with thermal state transitions.

Reads every results/device-pull/sustained-*.json and, for each, saves a
scatter of latency_ms vs elapsed_s with a rolling median overlay and
vertical lines marking thermal state transitions. Also saves a combined
chart overlaying the on-power and battery runs (RESULTS.md sessions 3 and
4) so the difference in thermal onset timing is visible on one axis.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

DEVICE_PULL_DIR = "results/device-pull"
CHARTS_DIR = "results/charts"
ROLLING_WINDOW = 101  # samples, centered

# RESULTS.md sessions 3 (on charger) and 4 (on battery) identify these runs
# by filename. Note: the 'notes' field inside both JSON files says
# "off power" for both, which contradicts RESULTS.md -- the labels below
# follow RESULTS.md as the source of truth; flag this if it matters. Both
# predate schema_version and its structured conditions (see
# format_conditions below), so this is exactly the kind of asserted-not-read
# mislabeling that schema 2 exists to catch.
RUN_LABELS = {
    "sustained-1788785293": "on-power",
    "sustained-1788866056": "battery",
}

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
    label = RUN_LABELS.get(os.path.splitext(os.path.basename(path))[0], "")
    title = f"{meta['device']} · {meta['model']} · {meta['precision']}"
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

    if len(runs) >= 2:
        out_path = os.path.join(CHARTS_DIR, "sustained_comparison.png")
        plot_comparison(runs, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
