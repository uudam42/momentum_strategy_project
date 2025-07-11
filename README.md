
# ML_SmartMomentum

This repository presents **ML_SmartMomentum**, a machine learning-enhanced momentum trading framework that integrates classic technical indicators with Random Forest classification. The system covers end-to-end modules: data acquisition, preprocessing, feature engineering, predictive modeling, backtesting, and performance evaluation.

---

## Project Overview

ML_SmartMomentum explores the intersection of traditional momentum-based technical analysis and modern supervised learning. It aims to forecast the next-day price direction of selected U.S. equities by training a classification model on derived indicators.

---

## Key Features

| Component            | Script / Module              | Description                                                                 |
|---------------------|------------------------------|-----------------------------------------------------------------------------|
| Data Acquisition     | `data_downloader.py`         | Pulls daily OHLCV stock data from Alpha Vantage for 5 liquid equities      |
| Feature Generation   | `features_generator.py`      | Computes RSI, EMA(20), and MACD indicators using pandas and numpy          |
| Model Training       | `machine_learning.py`        | Trains a Random Forest classifier to predict next-day price direction      |
| Signal Evaluation    | `ml_backtest.py`             | Generates trading signals and evaluates strategy vs. benchmark returns     |
| Result Visualization | `returns_comparison_log.png` | Shows cumulative return charts on both linear and logarithmic scales       |
| Report Document      | `ML_SmartMomentum.pdf`       | Full academic report with methodology and results                          |

---

## Dataset Description

- **Source**: Alpha Vantage API
- **Tickers Used**: AAPL, MSFT, GOOG, NVDA, TSLA
- **Time Span**: 2000 to 2025
- **Size**: Over 25,000 daily records across five equities
- **Cleaning Procedures**:
  - Handled missing values
  - Chronologically sorted records
  - Aligned data by date index

---

## Feature Engineering

Three momentum-based technical indicators were calculated:

- **RSI (Relative Strength Index)** – 14-day rolling
- **EMA20 (Exponential Moving Average)** – trend strength
- **MACD (Moving Average Convergence Divergence)** – captures crossover momentum

All indicators were implemented with `pandas` and `numpy`, preserving the temporal structure.

---

## Machine Learning Approach

- **Model**: RandomForestClassifier from `scikit-learn`
- **Input**: RSI, EMA20, MACD values
- **Target**: Binary variable indicating next-day price increase
- **Split**: 70% training, 30% testing
- **Accuracy**: Approximately 50.3% on test set
- **Model Storage**: Saved as `.pkl` in `features/model/` via `joblib`

---

## Backtesting Framework

- **Strategy Logic**: Buy if model predicts upward movement, hold otherwise
- **Return Computation**: Daily return multiplied by signal
- **Benchmark**: Equal-weighted portfolio of all five stocks
- **Visualization**: Uses `matplotlib` to compare cumulative returns
- **Outputs**: Stored in `images/` directory

---

## Technology Stack

- Python 3.10+
- `pandas`, `numpy` for data handling and indicators
- `scikit-learn` for machine learning
- `matplotlib` for visualization
- `alpha_vantage` for data acquisition
- `joblib` for model serialization

---

## Project Structure

```
ML_SmartMomentum/
├── features/
│   ├── raw_data/             # Cleaned stock data
│   ├── features_data/        # Processed indicators
│   ├── model/                # Trained RandomForest model
│   ├── data_downloader.py
│   ├── features_generator.py
│   └── machine_learning.py
├── ml_backtest.py            # Strategy simulation
├── images/                   # Return comparison plots
├── ML_SmartMomentum.pdf      # Full documentation
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/uudam42/momentum_strategy_project.git
cd momentum_strategy_project
```

### 2. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Execute Pipeline
```bash
python features/data_downloader.py
python features/features_generator.py
python features/machine_learning.py
python ml_backtest.py
```

---

## Sample Output

![Cumulative Return Plot](images/returns_comparison_log.png)

The strategy exhibits modest outperformance over the benchmark, as illustrated on a logarithmic cumulative return chart.

---

## Documentation

See the full report: [`ML_SmartMomentum.pdf`](./ML_SmartMomentum.pdf)  
Includes:
- Dataset statistics
- Feature definitions
- Model training workflow
- Backtesting methodology
- Performance interpretation

---

## Future Work

- Add risk-adjusted metrics (Sharpe, Max Drawdown)
- Hyperparameter tuning with cross-validation
- Incorporate additional technical & macroeconomic indicators
- Test with alternative models (e.g., XGBoost, LSTM)
- Integrate transaction cost simulation

---

## License

MIT License – Open for academic and non-commercial use.

---

## Author

Developed by **uudam42**  
For research inquiries or extensions, please contact via GitHub.
