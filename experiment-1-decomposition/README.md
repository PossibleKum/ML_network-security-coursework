# Experiment 1: Time Series Decomposition

## Aim
Perform time series decomposition of historical network traffic data into trend, seasonal, residual, and observed components.

## Dataset
`network_traffic.csv`

Columns:
- `timestamp` - hourly timestamp
- `traffic` - network traffic volume

## Method
The experiment uses additive seasonal decomposition with a period of 24, representing a repeating daily pattern in hourly traffic.

## Software
Python 3.10+ recommended.

## Installation

```bash
pip install -r ../requirements.txt
```

## Run in VS Code

Open the `experiment-1-decomposition` folder in VS Code and run:

```bash
python decomposition.py
```

## Output
The program creates:

```text
decomposition.png
```

The graph contains:
1. Observed traffic
2. Trend
3. Seasonal component
4. Residual component

## Result interpretation
- **Trend:** long-term increase/decrease in traffic.
- **Seasonal:** repeated daily traffic pattern.
- **Residual:** irregular variation not explained by trend or seasonality.
