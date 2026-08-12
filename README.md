# Network Traffic Time Series Analysis

This repository contains two laboratory experiments based on historical network traffic:

1. **Time Series Decomposition** - separates traffic into observed, trend, seasonal, and residual components.
2. **ARIMA Forecasting for Potential DDoS Detection** - forecasts traffic and flags unusually high forecast values as potential high-traffic/DDoS periods.

## Project Structure

```text
network_traffic_time_series/
├── experiment-1-decomposition/
│   ├── network_traffic.csv
│   ├── decomposition.py
│   └── README.md
├── experiment-2-ddos-forecasting/
│   ├── network_traffic.csv
│   ├── ddos_forecasting.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Experiment 1

```bash
cd experiment-1-decomposition
python decomposition.py
```

## Run Experiment 2

```bash
cd ..\experiment-2-ddos-forecasting
python ddos_forecasting.py
```

## GitHub

From the project root:

```bash
git init
git add .
git commit -m "Add time series decomposition and DDoS forecasting experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the repository URL with your own GitHub repository URL.

## Notes

The included CSV is a reproducible synthetic laboratory dataset designed to demonstrate the methods. For a final academic submission using real network traffic, replace it with your permitted historical dataset while keeping the same column names.
