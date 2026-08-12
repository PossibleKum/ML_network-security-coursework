"""
Experiment 1: Time Series Decomposition of Network Traffic

This program decomposes historical network traffic into:
- Observed
- Trend
- Seasonal
- Residual

Run:
    python decomposition.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "network_traffic.csv"
OUTPUT_FILE = BASE_DIR / "decomposition.png"


def main() -> None:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

    if "timestamp" not in df.columns or "traffic" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'traffic' columns.")

    df = df.sort_values("timestamp").set_index("timestamp")
    series = df["traffic"].astype(float).asfreq("h")

    # Fill any missing hourly observations by time interpolation.
    series = series.interpolate(method="time").ffill().bfill()

    # 24-hour seasonality is appropriate for hourly network traffic.
    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=24,
        extrapolate_trend="freq"
    )

    fig = decomposition.plot()
    fig.set_size_inches(12, 9)
    fig.suptitle("Time Series Decomposition of Network Traffic", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.show()

    print("Decomposition completed successfully.")
    print(f"Input dataset : {DATA_FILE}")
    print(f"Output plot   : {OUTPUT_FILE}")
    print("\nComponents:")
    print("- Observed  : original network traffic")
    print("- Trend     : long-term movement")
    print("- Seasonal  : repeating hourly pattern")
    print("- Residual  : unexplained/random variation")


if __name__ == "__main__":
    main()
