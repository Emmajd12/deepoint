import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATH = Path("deepoint_log.txt")  # change this if your log file has a different name


def extract_probabilities(log_path: Path):
    """Read the log file and extract all prob_pointing values as floats."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    probs = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Look for text like: prob_pointing=0.0123456789
            if "prob_pointing=" in line:
                # Use a regular expression to grab the number after '='
                match = re.search(r"prob_pointing=([0-9]*\.?[0-9]+)", line)
                if match:
                    value = float(match.group(1))
                    probs.append(value)

    return probs


def summarize(probs):
    """Print basic statistics about the probabilities."""
    if not probs:
        print("No prob_pointing values found in the log file.")
        return

    print(f"Total frames with prob_pointing: {len(probs)}")
    print(f"Min probability:  {min(probs):.4f}")
    print(f"Max probability:  {max(probs):.4f}")
    print(f"Mean probability: {statistics.mean(probs):.4f}")
    if len(probs) > 1:
        print(f"Std dev:          {statistics.pstdev(probs):.4f}")

    # How many frames are likely "pointing" if we use different thresholds:
    for thresh in [0.2, 0.5, 0.8]:
        count = sum(p >= thresh for p in probs)
        print(
            f"Frames with prob_pointing >= {thresh:.1f}: "
            f"{count} ({100 * count / len(probs):.1f}%)"
        )


def make_plots(probs):
    """Make and save a time-series plot and histogram of prob_pointing."""
    if not probs:
        return

    # --- 1) Time-series: probability vs frame index ---
    plt.figure()
    plt.plot(range(len(probs)), probs)
    plt.xlabel("Frame index (approx.)")
    plt.ylabel("prob_pointing")
    plt.title("Pointing probability over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prob_pointing_timeseries.png", dpi=150)
    plt.close()

    # --- 2) Histogram: distribution of probabilities ---
    plt.figure()
    plt.hist(probs, bins=20, edgecolor="black")
    plt.xlabel("prob_pointing")
    plt.ylabel("Count")
    plt.title("Distribution of pointing probabilities")
    plt.tight_layout()
    plt.savefig("prob_pointing_histogram.png", dpi=150)
    plt.close()

    print("Saved plots:")
    print("  prob_pointing_timeseries.png")
    print("  prob_pointing_histogram.png")


def main():
    print(f"Reading log from: {LOG_PATH}")
    probs = extract_probabilities(LOG_PATH)
    summarize(probs)
    make_plots(probs)


if __name__ == "__main__":
    main()
