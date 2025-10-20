# FinSight: Predicting Bitcoin Price Movements Using Technical Indicators and Machine Learning

## Problem Statement and Motivation
Cryptocurrency markets, and Bitcoin in particular, are characterized by high volatility and complex short-term dynamics that attract both traders and researchers. Traditional technical analysis tools — such as moving averages, RSI, and MACD — are widely used to interpret price charts, but their effectiveness depends heavily on the trader’s experience and subjective interpretation.
As someone passionate about finance and data analysis, I am particularly interested in exploring how quantitative methods and machine learning can be applied to financial markets, especially within the context of Bitcoin.
The motivation behind FinSight is to create a fully data-driven model capable of learning from historical Bitcoin price data and predicting short-term price movements (up or down) using combinations of technical indicators.  
The project aims to bridge the gap between traditional technical analysis and machine learning by providing a reproducible, transparent framework that can quantitatively evaluate and test trading hypotheses.

## Planned Approach and Technologies
The project will use Python 3.11 and follow a modular design to ensure clarity, scalability, and maintainability.  
Main components include:
- Data Collection: Using the yfinance library to download historical Bitcoin (BTC-USD) price data.
- Feature Engineering: Computing common technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA, Stochastic Oscillator) with the pandas and ta libraries.
- Labeling: Generating binary target variables that represent future price direction (e.g., 1 for upward movement, 0 for downward).
- Modeling: Training and comparing several machine learning models such as Logistic Regression, Random Forest, and XGBoost using scikit-learn.
- Evaluation: Assessing model accuracy, precision, and recall; visualizing confusion matrices and performance curves with matplotlib.
- Visualization: (Optional) Building a small Streamlit dashboard showing Bitcoin charts with predicted buy/sell signals.

All code will be organized under src/, tested with pytest, formatted using black, and linted with flake8. Continuous Integration via GitHub Actions will ensure quality control and automated testing.

## Expected Challenges and Mitigation
- Noisy and volatile data: Addressed through smoothing and the use of multiple time-based indicators.
- Overfitting: Avoided through time-series cross-validation and regularization.
- Data leakage: Prevented by proper chronological train-test splits.
- Model interpretability: Enhanced by analyzing feature importance and comparing multiple algorithms.

## Success Criteria
The project will be considered successful if:
- The trained model consistently outperforms a random or naïve baseline, achieving an accuracy higher than 55 % on unseen Bitcoin price data.
- Results are reproducible through documented, modular code and a clear workflow.
- Visualizations effectively communicate model performance and predicted trends.
- The repository adheres to clean coding, documentation, and testing best practices.

## Stretch Goals (if time permits)
- Add a backtesting engine to simulate trading performance based on model predictions.
- Implement a simple LSTM neural network for sequence-based learning.
- Deploy a Streamlit web app for real-time Bitcoin price visualization and predictions.

---

In summary, FinSight will explore whether machine learning can identify meaningful trading signals in Bitcoin price data by combining statistical modeling and technical analysis, resulting in a transparent, testable, and extensible financial prediction system.

