
# Finsight: Using Machine Learning to Dynamically Select Investment Strategies Based on Market Regimes

## Project Category  
Data Science / Machine Learning applied to Finance  

## Problem Statement and Motivation  
Financial markets evolve through different regimes such as periods of high volatility, strong upward trends, or market corrections. Traditional investment strategies like buy-and-hold, momentum, and mean reversion perform differently under these conditions but are usually applied in a static way, without adapting to changing environments.  

The goal of this project is to build a machine learning model capable of identifying market regimes and predicting which investment strategy is likely to perform best in the near future. This approach aims to create an adaptive investment system that dynamically switches between strategies based on predicted market conditions.  

As someone passionate about finance, this project represents an opportunity to apply quantitative methods to portfolio management and explore how machine learning can enhance traditional investment logic.

## Planned Approach and Technologies  
The project will be implemented in Python 3.11 following a clear, modular, and reproducible structure.  

**Main steps include:**  
1. **Data Collection and Preparation**  
   - Retrieve historical market data (indices or ETFs) using the `yfinance` API.  
   - Compute financial indicators such as volatility, trend strength (RSI, MACD, SMA), and correlations between assets.
2. **Feature Engineering and Labeling**  
   - Create input features describing volatility regimes, trend indicators, correlation structures, and simple macroeconomic variables (e.g., VIX index, interest rate).  
   - Define the target variable as the investment strategy (momentum, mean reversion, or buy-and-hold) that achieved the best performance in the following time period.  

3. **Modeling**  
   - Train and compare several models, including Logistic Regression, Random Forest, and Gradient Boosting.  
   - Use time-series cross-validation to prevent data leakage and ensure realistic evaluation.  

4. **Dynamic Backtesting**  
   - Use model predictions to dynamically select the strategy to apply at each time step.  
   - Compare the ML-based adaptive portfolio to static strategies in terms of cumulative return, volatility, and Sharpe ratio.  

## Expected Challenges and How They’ll Be Addressed  
- **Overfitting:** Controlled through regularization and time-based validation.  
- **Noisy financial data:** Mitigated by smoothing indicators and selecting robust features.  
- **Interpretability:** Improved by analyzing feature importance and comparing model behavior across market conditions.  

## Success Criteria  
The project will be considered successful if:  
- The adaptive ML model outperforms static strategies during backtesting.  
- Results are reproducible and supported by clear visualizations.  
- The repository follows good coding practices and documentation standards.  

## Stretch Goals (if time permits)  
- Extend the model to include multiple asset classes such as bonds or cryptocurrencies.  
- Develop a simple Streamlit dashboard to visualize detected market regimes and strategy decisions in real time.  

---

In summary, Finsight combines financial analysis and machine learning to build an adaptive portfolio management system capable of identifying market conditions and selecting the most appropriate investment strategy dynamically, bridging traditional financial intuition with data-driven decision-making.

