# Finsight — Machine Learning for Adaptive Asset Allocation

Finsight is an automated machine learning pipeline that dynamically allocates capital across three classical strategies — **Buy & Hold**, **Momentum**, and **Mean Reversion** — based on predicted future market regimes.  
The project includes **SP500 monthly analysis**, **BTC monthly**, and **BTC weekly** pipelines, each with **walk-forward ML forecasting**, full **backtesting**, and **auto-generated plots**.

---

## Installation

### Using Conda
conda env create -f environment.yml
conda activate finsight-env


## Run the Full Project

Simply run:
python main.py

Runtime note: the full pipeline may take several minutes to execute 
due to walk-forward backtesting and model training.



This executes automatically:

1. **SP500 monthly pipeline**  
2. **BTC monthly pipeline**  
3. **BTC weekly pipeline**  
4. **Generation of figures (equity curves, bar charts, performance summaries)**

All results are exported to:

- `data/results/`
- `fig/`

---

## Project Structure

DSAP-Project/
├── main.py
├── README.md
├── Proposal.md
├── environment.yml
│
├── src/
│   ├── main.py
│   │
│   ├── data_pipeline/
│   │   ├── prepare_sp500.py
│   │   ├── prepare_btc_monthly.py
│   │   ├── prepare_btc_weekly.py
│   │
│   ├── modeling/
│   │   ├── modeling_sp500.py
│   │   ├── modeling_btc_monthly.py
│   │   ├── modeling_btc_weekly.py
│   │
│   ├── backtesting/
│   │   ├── backtesting.py
│   │   ├── backtesting_btc.py
│   │   ├── backtesting_btc_weekly.py
│   │
│   └── plotting/
│       ├── plot_results.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
└── fig/


---

## Methodology Overview

### 1. Data Processing  
- Clean and align raw SPY and BTC price data  
- Compute technical features (returns, volatility, SMA ratios, RSI, MACD, z-scores)  
- Aggregate daily BTC data into weekly features  
- Create labels: **best strategy over next horizon**

### 2. Modeling  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Rolling-window walk-forward evaluation  
- Time-series aware splits with expanding training window

### 3. Backtesting  
Each strategy is evaluated on:  
- Annualized return  
- Annualized volatility  
- Sharpe ratio  
- Maximum drawdown  
- Equity curve  
- Monthly returns  

### 4. ML Adaptive Strategy  
Chooses each month/week the strategy predicted to outperform.

---

## Results Summary

### **SP500 Monthly (1985–2025)**

| Strategy                          | Ann. Return | Sharpe | Max DD   |
|----------------------------------|-------------|--------|----------|
| Buy & Hold                       | 8.29%       | 0.55   | -50.78%  |
| Momentum                         | 5.10%       | 0.51   | -30.25%  |
| Mean Reversion                   | 3.09%       | 0.27   | -40.86%  |
| ML Adaptive (BH by default)      | 3.42%       | 0.30   | -43.01%  |
| Oracle (best of 3, ex post)      | 28.08%      | 3.18   | 0%       |


ML allocation distribution:  
- Buy & Hold: **65.48%**  
- Mean Reversion: **20.32%**  
- Momentum: **14.19%**

---

### **BTC Monthly**

| Strategy        | Ann. Return | Sharpe | Max DD |
|----------------|-------------|--------|--------|
| Buy & Hold     | 63.32%      | 0.82   | -75.57% |
| Momentum       | 52.30%      | 0.83   | -64.48% |
| Mean Reversion | 7.24%       | 0.15   | -60.52% |
| ML Adaptive    | 177.36%     | 2.73   | -45.85% |
| Oracle         | 243.35%     | 4.15   | 0% |

---

### **BTC Weekly**

| Strategy        | Ann. Return | Sharpe | Max DD |
|----------------|-------------|--------|--------|
| Buy & Hold     | 53.18%      | 0.76   | -81.69% |
| Momentum       | 66.27%      | 1.21   | -63.41% |
| Mean Reversion | 14.76%      | 0.34   | -64.38% |
| ML Adaptive    | 82.51%      | 1.40   | -53.88% |
| Oracle         | 488.07%     | 9.40   | -17.13% |

---

## Output Files

Generated in `data/results/`:

- `performance_summary_*.csv`
- `monthly_returns_*.csv`
- `equity_curves_*.csv`

Generated in `fig/`:

- `sp500_equity_curves.png`
- `btc_monthly_equity_curves.png`
- `btc_weekly_equity_curves.png`
- Annualized return bar charts  
- Sharpe ratio charts

---

## Reproducibility

- All random seeds fixed (`random_state`)  
- Raw input datasets (SPY and BTC price series) are included in `data/raw/`.
- Deterministic pipelines  
- `environment.yml` provided  
- Running `python main.py` re-creates **all datasets, all results and all plots**

---

## Author

**Sofian Ben Yedder**  
Master in Finance — HEC Lausanne  
DSAP Project (Fall 2025)

