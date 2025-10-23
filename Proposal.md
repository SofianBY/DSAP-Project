# Finsight: Testing and Comparing Investment Strategies with Backtesting

## Problem Statement and Motivation
In financial markets, investors use various strategies, from passive "buy and hold" investing to more active, rule-based approaches such as momentum or mean reversion. While these strategies are often discussed in theory, few individual investors have the tools to systematically test and compare their real performance over time.

As someone passionate about finance and investment, I aim to build a data-driven framework that evaluates multiple investment strategies using real market data. The goal of Finsight is not to predict prices but to determine which types of strategies perform best under different market conditions. By backtesting and comparing results, the project will highlight the strengths and weaknesses of common quantitative approaches and bridge the gap between financial theory and real-world performance.

## Planned Approach and Technologies
The project will be implemented in Python 3.11 with a modular structure for clarity and reproducibility.  
Main components include:

- Data Collection: Using the yfinance library to download historical daily prices for a diversified set of stocks (for example, companies from the S&P 500) to evaluate performance across sectors.  
- Feature Engineering: Calculating returns, moving averages, and volatility with pandas and numpy.  
- Strategy Implementation: Coding several investment strategies:
  - Buy and Hold (benchmark)
  - Momentum (buy recent winners)
  - Mean Reversion (buy recent losers)
  - Moving Average Crossover (technical trading rule)
- Backtesting Engine: Simulating each strategy’s performance over time, reinvesting gains and tracking portfolio evolution.  
- Evaluation Metrics: Comparing annualized return, volatility, Sharpe ratio, and maximum drawdown.  
- Visualization: Plotting equity curves, risk/return comparisons, and heatmaps of performance using matplotlib or plotly.

Instead of focusing on a few individual assets, each strategy will be applied to a broader universe of stocks to enable a robust comparison and identify which approaches perform consistently across market environments.

All code will be tested with pytest, formatted using black, and linted with flake8. Continuous Integration via GitHub Actions will ensure code quality and reproducibility.

## Expected Challenges and Mitigation
- Data quality and missing values: Managed through preprocessing and validation.  
- Look-ahead bias: Avoided by careful time-based simulation logic.  
- Overfitting: Controlled by limiting strategy parameters and using realistic assumptions.  
- Interpretability: Ensured through clear performance metrics and visualization.

## Success Criteria
The project will be considered successful if:
- At least three investment strategies are implemented, tested, and compared on real data.  
- Results are reproducible and supported by visualizations.  
- The analysis identifies conditions where each strategy performs best.  
- The repository follows clean coding and documentation standards.

## Stretch Goals (if time permits)
- Add simple machine learning to detect favorable market regimes.  
- Combine strategies to compare portfolio-level performance.  
- Build a Streamlit dashboard for interactive backtesting and visualization.

---

In summary, Finsight provides a transparent and educational framework for understanding how different investment strategies perform in practice. Rather than predicting prices, the project uses historical data to evaluate and compare systematic approaches in a rigorous, data-driven way.
