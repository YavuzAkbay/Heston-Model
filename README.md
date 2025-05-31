# 🧠 Enhanced Heston Model with Deep Learning for Stock Price Prediction

A financial modeling system that combines the Heston stochastic volatility model with deep neural networks to provide enhanced stock price forecasting and volatility prediction. This project leverages LSTM networks with attention mechanisms and differential neural networks to calibrate Heston model parameters and generate accurate market predictions.

## 🚀 Features

- Dual Neural Network Architecture: Separate networks for parameter calibration and volatility forecasting
- Heston Model Implementation: Advanced stochastic volatility modeling with mean reversion
- Deep Differential Networks: ML-powered parameter estimation for kappa, theta, xi, rho, and v0
- LSTM with Attention: Multi-head attention mechanism for volatility pattern recognition
- Monte Carlo Simulations: Enhanced path generation with ML-calibrated parameters
- Comprehensive Risk Analysis: VaR, CVaR, and probability-based investment insights
- Real-time Calibration: Dynamic parameter adjustment based on market conditions
- Advanced Visualizations: Multi-panel analysis with confidence intervals and distributions

## 📊 Model Architecture

- DeepDifferentialHeston Network
  - Purpose: Calibrates Heston model parameters (κ, θ, ξ, ρ, v₀)
  - Architecture: Deep feedforward network with ELU activation and batch normalization
  - Input: Market features and historical data
  - Output: Optimized Heston parameters
- VolatilityForecastingNetwork
  - Purpose: Predicts future volatility patterns
  - Architecture: LSTM with multi-head attention mechanism
  - Input: Sequential volatility features and market indicators
  - Output: Future volatility estimates

### Key Components

1. **LSTM Layers**: Capture sequential patterns in financial time series
2. **Multi-head Attention**: Focus on relevant historical periods
3. **Feature Extraction**: Dense layers for complex pattern recognition
4. **Parameter Prediction**: Separate heads for price, volatility, and drift

## 🛠 Installation

1. Clone the repository

```bash
git clone https://github.com/YavuzAkbay/Heston-Model
cd Heston-Model
```

2. Install required packages

```bash
pip install -r requirements.txt
```

3. For GPU acceleration (optional):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## 📈 Usage

### Basic Usage

```python
from heston_analysis import analyze_stock_with_heston, create_visualizations

# Analyze a stock with traditional Heston model
symbol = "AAPL"
stock_data, S_paths, v_paths, heston_model = analyze_stock_with_heston(
    symbol, 
    forecast_months=6
)

# Create comprehensive visualizations
create_visualizations(symbol, stock_data, S_paths, v_paths, heston_model)
```

### Enhanced ML Analysis

```python
from enhanced_heston_ml import run_enhanced_analysis

# Run complete ML-enhanced analysis
enhanced_heston, S_paths, v_paths = run_enhanced_analysis(
    symbol="TSLA",
    forecast_months=12
)
```

### Custom Model Configuration

```python
from enhanced_heston_ml import EnhancedHestonML

# Initialize with custom parameters
model = EnhancedHestonML(
    S0=current_price,
    r=risk_free_rate,
    T=time_horizon
)

# Train with custom epochs
model.train_networks(
    stock_data,
    epochs_param=150,
    epochs_vol=300
)

# Generate forecasts
S_paths, v_paths, predicted_vol = model.enhanced_forecast(
    stock_data,
    forecast_months=6,
    n_simulations=5000
)
```

## 🔬 Heston Model Parameters

### Core Parameters
- κ (kappa): Mean reversion speed of volatility
- θ (theta): Long-term variance level
- ξ (xi): Volatility of volatility (vol-of-vol)
- ρ (rho): Correlation between price and volatility processes
- v₀: Initial variance level

### ML Enhancement Features
- Dynamic Calibration: Parameters adjust based on market regime
- Pattern Recognition: LSTM networks identify volatility clustering
- Attention Mechanism: Focus on relevant historical periods
- Regime Detection: Automatic adaptation to market conditions

## 📊 Model Output

### Comprehensive Analysis Dashboard
1. Price Path Simulations: Monte Carlo paths with confidence intervals
2. Volatility Evolution: Stochastic volatility forecasting
3. Training Progress: Neural network convergence monitoring
4. Return Distributions: Risk analysis with VaR calculations

### Investment Insights
- Probability of Profit: Statistical likelihood of positive returns
- Expected Returns: ML-enhanced vs traditional estimates
- Risk Metrics: VaR, CVaR, and Sharpe ratio analysis
- Volatility Forecasts: Dynamic volatility predictions
- Model Confidence: ML model reliability scores

## 🧮 Mathematical Foundation

The Heston model is governed by the following stochastic differential equations:

Price Process:
`dS(t) = rS(t)dt + √v(t)S(t)dW₁(t)`

Volatility Process:
`dv(t) = κ(θ - v(t))dt + ξ√v(t)dW₂(t)`

Where dW₁(t) and dW₂(t) are correlated Brownian motions with correlation ρ.

## 🎯 Performance Features

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Gradient Clipping**: Stable training with norm clipping
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Early Stopping**: Patience-based training termination
- **Batch Normalization**: Improved convergence and stability

## 🔧 Configuration Options

### Model Parameters
```python
# DeepDifferentialHeston configuration
input_dim = 5           # Number of input features
hidden_dims = [128, 256, 128, 64]  # Network architecture
dropout = 0.1           # Dropout rate

# VolatilityForecastingNetwork configuration
input_size = 10         # Sequence input size
hidden_size = 128       # LSTM hidden units
num_layers = 3          # LSTM depth
num_heads = 8           # Attention heads
```

### Training Parameters
```python
epochs_param = 100      # Parameter network training epochs
epochs_vol = 200        # Volatility network training epochs
batch_size = 512        # Training batch size
learning_rate = 0.001   # Initial learning rate
weight_decay = 1e-5     # L2 regularization
```

### Simulation Parameters
```python
n_simulations = 5000    # Monte Carlo paths
n_steps = 252           # Steps per year
forecast_months = 6     # Prediction horizon
```


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GPL v3 - see the (https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. **Do not use this for actual trading decisions without proper risk management and professional financial advice.** Past performance does not guarantee future results. Trading stocks involves substantial risk of loss.

## 🙏 Acknowledgments

- **Steven Heston**: Original Heston stochastic volatility model (1993)
- **PyTorch**: Deep learning framework
- **yfinance**: Financial data API
- **Financial Mathematics Community**: Stochastic calculus foundations

## 📧 Contact

Yavuz Akbay - akbay.yavuz@gmail.com

## 📚 References

1. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
2. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
3. Rouah, F. D. (2013). "The Heston Model and Its Extensions in Matlab and C#"

---

⭐️ If this project helped with your financial analysis, please consider giving it a star!

**Built with ❤️ for the intersection of mathematics, machine learning, and finance**
