# Finsight — Machine Learning for Adaptive Asset Allocation

Finsight is an automated machine learning pipeline that dynamically allocates capital across three classical strategies — Buy & Hold, Momentum, and Mean Reversion — based on predicted future market regimes.

The project includes:
- SP500 monthly analysis
- BTC monthly analysis
- BTC weekly analysis

Each pipeline uses walk-forward machine learning forecasting, full backtesting, and auto-generated performance figures.

---

## Installation

Using Conda:

conda env create -f environment.yml  
conda activate finsight-env

---

## Run the Full Project

Simply run:

python main.py

Runtime note: the full pipeline may take several minutes to execute due to walk-forward backtesting and model training.

This automatically executes:
1. SP500 monthly pipeline  
2. BTC monthly pipeline  
3. BTC weekly pipeline  
4. Generation of figures (equity curves, bar charts, performance summaries)

All outputs are saved to:
- data/results/
- fig/

---

## Project Structure

DSAP-Project/  
├── main.py  
├── README.md  
├── Proposal.md  
├── environment.yml  
├── src/  
│   ├── data_pipeline/  
│   │   ├── prepare_sp500.py  
│   │   ├── prepare_btc_monthly.py  
│   │   └── prepare_btc_weekly.py  
│   ├── modeling/  
│   │   ├── modeling_sp500.py  
│   │   ├── modeling_btc_monthly.py  
│   │   └── modeling_btc_weekly.py  
│   ├── backtesting/  
│   │   ├── backtesting.py  
│   │   ├── backtesting_btc.py  
│   │   └── backtesting_btc_weekly.py  
│   └── plotting/  
│       └── plot_results.py  
├── data/  
│   ├── raw/  
│   └── processed/  
├── results/  
└── fig/

---

## Methodology Overview

### Data Processing
- Clean and align raw SPY and BTC price data
- Compute technical features (returns, volatility, SMA ratios, RSI, MACD, z-scores)
- Aggregate daily BTC data into weekly features
- Create labels: best strategy over the next horizon

### Modeling
- Logistic Regression
- Random Forest
- Gradient Boosting
- Rolling-window walk-forward evaluation
- Time-series aware splits with expanding training windows

### Backtesting
Each strategy is evaluated using:
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Equity curves
- Monthly and weekly returns

### ML Adaptive Strategy
At each period, the model selects the strategy predicted to outperform.

---
## Results Summary

### SP500 Monthly (1985–2025)

| Strategy            | Annualized Return | Sharpe | Max Drawdown |
|---------------------|------------------:|-------:|-------------:|
| Buy & Hold          | 8.29%             | 0.55   | -50.78%      |
| Momentum            | 5.10%             | 0.51   | -30.25%      |
| Mean Reversion      | 3.09%             | 0.27   | -40.86%      |
| ML Adaptive         | 3.42%             | 0.30   | -43.01%      |
| Oracle (ex post)    | 28.08%            | 3.18   | 0.00%        |

---

### BTC Monthly

| Strategy            | Annualized Return | Sharpe |
|---------------------|------------------:|-------:|
| Buy & Hold          | 63.32%            | 0.82   |
| Momentum            | 52.30%            | 0.83   |
| Mean Reversion      | 7.24%             | 0.15   |
| ML Adaptive         | 177.36%           | 2.73   |
| Oracle              | 243.35%           | 4.15   |

---

### BTC Weekly

| Strategy            | Annualized Return | Sharpe |
|---------------------|------------------:|-------:|
| Buy & Hold          | 53.18%            | 0.76   |
| Momentum            | 66.27%            | 1.21   |
| Mean Reversion      | 14.76%            | 0.34   |
| ML Adaptive         | 82.51%            | 1.40   |
| Oracle              | 488.07%           | 9.40   |





---

## Reproducibility

- All random seeds fixed (random_state)
- Raw input datasets included in data/raw/
- Deterministic pipelines
- environment.yml provided
- Running python main.py fully reproduces all datasets, results, and figures

---

## Author

Sofian Ben Yedder  
Master in Finance — HEC Lausanne  
DSAP Project (Fall 2025)


