"""
Experiment 2: Network Traffic Forecasting for Potential DDoS Detection

This program:
1. Loads historical network traffic.
2. Splits it into training and testing data.
3. Fits an ARIMA model.
4. Forecasts the test period.
5. Calculates an anomaly threshold from the training data.
6. Flags forecast values above the threshold as potential high-traffic/DDoS periods.

Important:
A forecast threshold is only a demonstration of anomaly detection.
It is not proof that a DDoS attack has occurred.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "network_traffic.csv"
PLOT_FILE = BASE_DIR / "forecast.png"
RESULT_FILE = BASE_DIR / "forecast_results.csv"


def main() -> None:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

    if "timestamp" not in df.columns or "traffic" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'traffic' columns.")

    df = df.sort_values("timestamp").set_index("timestamp")
    series = df["traffic"].astype(float).asfreq("h")
    series = series.interpolate(method="time").ffill().bfill()

    # Hold out the final 20% for testing.
    split_index = int(len(series) * 0.80)
    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    if len(test) == 0:
        raise ValueError("Dataset is too short for train/test forecasting.")

    # ARIMA(p,d,q)
    # This is a simple lab-friendly starting model.
    model = ARIMA(train, order=(2, 1, 2))
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test))
    forecast.index = test.index

    # Training-based anomaly threshold.
    # A high forecast relative to the training distribution is flagged.
    threshold = train.mean() + 3 * train.std()

    results = pd.DataFrame({
        "timestamp": test.index,
        "actual_traffic": test.values,
        "forecast_traffic": forecast.values,
    })
    results["potential_ddos"] = results["forecast_traffic"] > threshold
    results["threshold"] = threshold
    results.to_csv(RESULT_FILE, index=False)

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))

    plt.figure(figsize=(13, 6))
    plt.plot(train.index, train, label="Training traffic")
    plt.plot(test.index, test, label="Actual traffic")
    plt.plot(forecast.index, forecast, label="ARIMA forecast", linestyle="--")
    plt.axhline(
        threshold,
        linestyle=":",
        linewidth=2,
        label=f"Potential DDoS threshold ({threshold:.1f})"
    )

    flagged = results[results["potential_ddos"]]
    if not flagged.empty:
        plt.scatter(
            flagged["timestamp"],
            flagged["forecast_traffic"],
            marker="x",
            s=60,
            label="Potential DDoS / high-traffic forecast"
        )

    plt.title("ARIMA Forecasting of Network Traffic")
    plt.xlabel("Time")
    plt.ylabel("Network Traffic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.show()

    print("ARIMA forecasting completed successfully.")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Potential DDoS/high-traffic forecast periods: {flagged.shape[0]}")
    print(f"\nForecast plot : {PLOT_FILE}")
    print(f"Results CSV   : {RESULT_FILE}")
    print(
        "\nNote: This experiment demonstrates forecasting-based anomaly detection. "
        "A real DDoS detection system should combine traffic features, security logs, "
        "and a validated detection model."
    )


if __name__ == "__main__":
    main()
