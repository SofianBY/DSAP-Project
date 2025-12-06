```markdown
# Finsight – Market Regime Adaptive Strategies

This project investigates how different market regimes influence the performance of three classic investment strategies: Buy & Hold, Momentum, and Mean Reversion. A machine learning model predicts which strategy is likely to perform best in the following month and dynamically switches between them.

## Installation
```bash
conda env create -f environment.yml
conda activate finsight-env
```

## Run the full pipeline
```bash
python -m src.main
```

## Project Structure
DSAP-Project/
├─ README.md  
├─ Proposal.md  
├─ environment.yml  
├─ src/  
│   ├─ data_download.py  
│   ├─ features.py  
│   ├─ labeling.py  
│   ├─ enrich_features.py  
│   ├─ modeling.py  
│   ├─ modeling_monthly.py  
│   ├─ backtesting.py  
│   ├─ save_results.py  
│   ├─ plot_results.py  
│   └─ main.py  
└─ data/ (raw, processed, results auto-generated)

## Pipeline Steps
1. Download SPY price data  
2. Compute daily features (returns, volatility, RSI, MACD, SMA…)  
3. Label the future monthly best-performing strategy  
4. Add macroeconomic series (VIX, yields)  
5. Build monthly features  
6. Train ML models (Logistic Regression, Random Forest, Gradient Boosting)  
7. Backtest BH, MOM, MR, ML Adaptive, Oracle  
8. Save CSV results + plots  

## Results Summary (Nov 2025)
| Strategy | Ann. Return | Sharpe | Max DD |
|----------|-------------|--------|--------|
| Buy & Hold | 8.29% | 0.55 | -50.78% |
| Momentum | 5.10% | 0.51 | -30.25% |
| Mean Reversion | 3.09% | 0.27 | -40.86% |
| ML Adaptive | 3.42% | 0.30 | -43.01% |
| Oracle | 28.08% | 3.18 | 0% |

ML chooses per test-month:  
- BH: 32.0%  
- MOM: 23.6%  
- MR: 44.4%

## Output Files
Saved to data/results/ :
- performance_summary.csv  
- equity_curves.csv  
- monthly_returns.csv  
- equity_curves.png  
- annualized_return_bar.png  
- sharpe_ratio_bar.png  

## Reproducibility
- All seeds fixed  
- Full pipeline automated  
- environment.yml provided  

## Stretch Goals
- Multi-asset extension  
- Streamlit dashboard for real-time regime predictions  

## Author
Sofian Ben Yedder  
HEC Lausanne – DSAP Project (Fall 2025)
```
