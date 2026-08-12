# Experiment 2: Network Traffic Forecasting for Potential DDoS Detection

## Aim
Apply a time series forecasting method (ARIMA) to historical network traffic and identify unusually high forecasted traffic as potential DDoS/high-traffic periods.

## Dataset
`network_traffic.csv`

Columns:
- `timestamp` - hourly timestamp
- `traffic` - network traffic volume

## Method
The experiment uses an ARIMA(2,1,2) model.

The final 20% of observations are used as a test set. The forecast is compared with the actual traffic.

A demonstration threshold is calculated as:

```text
threshold = mean(training traffic) + 3 × standard deviation(training traffic)
```

Forecast values above this threshold are flagged as potential high-traffic/DDoS periods.

## Installation

```bash
pip install -r ../requirements.txt
```

## Run in VS Code

Open the `experiment-2-ddos-forecasting` folder in VS Code and run:

```bash
python ddos_forecasting.py
```

## Output
The program creates:

```text
forecast.png
forecast_results.csv
```

## Evaluation
The program reports:
- MAE
- RMSE
- Number of flagged potential high-traffic periods

## Important limitation
This is a forecasting/anomaly-detection laboratory experiment. A high forecast does not prove that a DDoS attack will happen or has happened. A real security system should use validated attack labels and additional network/security features.
