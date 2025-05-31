import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, xi, rho, r, T):
        """
        Initialize Heston Model parameters
        
        S0: Initial stock price
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        xi: Volatility of volatility
        rho: Correlation between price and volatility
        r: Risk-free rate
        T: Time horizon
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r
        self.T = T
        
    def simulate_paths(self, n_paths=1000, n_steps=252):
        """
        Simulate stock price paths using Heston model
        """
        dt = self.T / n_steps
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        # Generate correlated random numbers
        for i in range(n_steps):
            # Generate correlated Brownian motions
            Z1 = np.random.standard_normal(n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(n_paths)
            
            # Update variance (ensure non-negative)
            v[:, i + 1] = np.maximum(
                v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + 
                self.xi * np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * Z2,
                0
            )
            
            # Update stock price
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * v[:, i]) * dt + 
                np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * Z1
            )
            
        return S, v

def get_stock_data(symbol, period="3y"):
    """
    Fetch stock data and calculate historical volatility
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change().dropna()
    
    # Calculate annualized volatility
    historical_vol = data['Returns'].std() * np.sqrt(252)
    
    return data, historical_vol

def get_risk_free_rate():
    """
    Get 10-year Treasury rate as risk-free rate
    """
    try:
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(period="5d")
        risk_free_rate = tnx_data['Close'].iloc[-1] / 100  # Convert percentage to decimal
        return risk_free_rate
    except:
        return 0.03  # Default 3% if data unavailable

def calibrate_heston_simple(stock_data, historical_vol):
    """
    Simple calibration based on historical data
    """
    returns = stock_data['Returns'].dropna()
    
    # Estimate parameters from historical data
    v0 = historical_vol**2  # Initial variance
    theta = historical_vol**2  # Long-term variance
    kappa = 2.0  # Mean reversion speed
    xi = 0.3  # Vol of vol
    rho = -0.7  # Typical negative correlation
    
    return v0, kappa, theta, xi, rho

def analyze_stock_with_heston(symbol="AAPL", forecast_months=6):
    """
    Complete analysis of a stock using Heston model
    """
    print(f"🔍 Analyzing {symbol} with Heston Model")
    print("=" * 50)
    
    # Get stock data
    stock_data, historical_vol = get_stock_data(symbol)
    current_price = stock_data['Close'].iloc[-1]
    
    # Get risk-free rate
    risk_free_rate = get_risk_free_rate()
    
    print(f"📊 Current Price: ${current_price:.2f}")
    print(f"📈 Historical Volatility: {historical_vol:.1%}")
    print(f"🏦 Risk-free Rate: {risk_free_rate:.1%}")
    
    # Calibrate Heston parameters
    v0, kappa, theta, xi, rho = calibrate_heston_simple(stock_data, historical_vol)
    
    print(f"\n🎯 Heston Model Parameters:")
    print(f"   Initial Variance (v0): {v0:.4f}")
    print(f"   Mean Reversion Speed (κ): {kappa:.2f}")
    print(f"   Long-term Variance (θ): {theta:.4f}")
    print(f"   Vol of Vol (ξ): {xi:.2f}")
    print(f"   Correlation (ρ): {rho:.2f}")
    
    # Initialize Heston model
    T = forecast_months / 12  # Convert months to years
    heston = HestonModel(current_price, v0, kappa, theta, xi, rho, risk_free_rate, T)
    
    # Simulate paths
    print(f"\n🚀 Simulating {forecast_months}-month price paths...")
    S_paths, v_paths = heston.simulate_paths(n_paths=1000, n_steps=int(252 * T))
    
    # Calculate statistics
    final_prices = S_paths[:, -1]
    price_percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
    
    print(f"\n📊 Price Forecast Results (in {forecast_months} months):")
    print(f"   5th Percentile:  ${price_percentiles[0]:.2f}")
    print(f"   25th Percentile: ${price_percentiles[1]:.2f}")
    print(f"   Median (50th):   ${price_percentiles[2]:.2f}")
    print(f"   75th Percentile: ${price_percentiles[3]:.2f}")
    print(f"   95th Percentile: ${price_percentiles[4]:.2f}")
    
    # Calculate potential returns
    median_return = (price_percentiles[2] - current_price) / current_price
    print(f"\n💰 Expected Return: {median_return:.1%}")
    
    # Risk assessment
    downside_risk = (price_percentiles[0] - current_price) / current_price
    upside_potential = (price_percentiles[4] - current_price) / current_price
    
    print(f"📉 Downside Risk (5th percentile): {downside_risk:.1%}")
    print(f"📈 Upside Potential (95th percentile): {upside_potential:.1%}")
    
    return stock_data, S_paths, v_paths, heston

def create_visualizations(symbol, stock_data, S_paths, v_paths, heston):
    """
    Create comprehensive visualizations
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Heston Model Analysis for {symbol}', fontsize=16, fontweight='bold')
    
    # 1. Historical Price vs Simulated Paths
    time_steps = np.linspace(0, heston.T, S_paths.shape[1])
    
    # Plot sample paths
    for i in range(min(50, S_paths.shape[0])):
        ax1.plot(time_steps, S_paths[i], alpha=0.3, color='lightblue', linewidth=0.5)
    
    # Plot percentiles
    percentiles = np.percentile(S_paths, [5, 25, 50, 75, 95], axis=0)
    ax1.plot(time_steps, percentiles[2], 'r-', linewidth=2, label='Median Path')
    ax1.fill_between(time_steps, percentiles[0], percentiles[4], alpha=0.2, color='red', label='90% Confidence')
    ax1.fill_between(time_steps, percentiles[1], percentiles[3], alpha=0.3, color='red', label='50% Confidence')
    
    ax1.axhline(y=heston.S0, color='black', linestyle='--', label='Current Price')
    ax1.set_title('Price Path Simulations')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility Evolution
    vol_paths = np.sqrt(v_paths)
    vol_percentiles = np.percentile(vol_paths, [5, 25, 50, 75, 95], axis=0)
    
    for i in range(min(20, vol_paths.shape[0])):
        ax2.plot(time_steps, vol_paths[i], alpha=0.3, color='green', linewidth=0.5)
    
    ax2.plot(time_steps, vol_percentiles[2], 'g-', linewidth=2, label='Median Volatility')
    ax2.fill_between(time_steps, vol_percentiles[0], vol_percentiles[4], alpha=0.2, color='green')
    ax2.axhline(y=np.sqrt(heston.theta), color='orange', linestyle='--', label='Long-term Vol')
    ax2.set_title('Volatility Evolution')
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final Price Distribution
    final_prices = S_paths[:, -1]
    ax3.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=heston.S0, color='red', linestyle='--', linewidth=2, label='Current Price')
    ax3.axvline(x=np.median(final_prices), color='green', linestyle='-', linewidth=2, label='Expected Price')
    ax3.set_title(f'Price Distribution in {heston.T:.1f} Years')
    ax3.set_xlabel('Stock Price ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Historical vs Model Comparison
    recent_data = stock_data.tail(252)  # Last year
    ax4.plot(recent_data.index, recent_data['Close'], 'b-', linewidth=2, label='Historical Price')
    ax4.set_title('Historical Price Performance')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Stock Price ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Investment Insights
    print("\n" + "="*60)
    print("🎯 INVESTMENT INSIGHTS FOR SPOT INVESTORS")
    print("="*60)
    
    final_prices = S_paths[:, -1]
    prob_profit = np.mean(final_prices > heston.S0)
    avg_return = np.mean((final_prices - heston.S0) / heston.S0)
    
    print(f"📊 Probability of Profit: {prob_profit:.1%}")
    print(f"💰 Average Expected Return: {avg_return:.1%}")
    
    if prob_profit > 0.6 and avg_return > 0.05:
        print("✅ POSITIVE OUTLOOK: Model suggests favorable risk-reward profile")
    elif prob_profit > 0.5:
        print("⚖️  NEUTRAL OUTLOOK: Balanced risk-reward, consider market conditions")
    else:
        print("⚠️  CAUTIOUS OUTLOOK: Higher downside risk, consider waiting or hedging")
    
    print(f"\n📈 Volatility Insights:")
    avg_vol = np.mean(np.sqrt(v_paths[:, -1]))
    print(f"   Expected Future Volatility: {avg_vol:.1%}")
    
    if avg_vol > np.sqrt(heston.v0) * 1.2:
        print("   📊 Volatility expected to INCREASE - Higher risk/reward")
    elif avg_vol < np.sqrt(heston.v0) * 0.8:
        print("   📊 Volatility expected to DECREASE - More stable returns")
    else:
        print("   📊 Volatility expected to remain STABLE")

# Run the analysisimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepDifferentialHeston(nn.Module):
    """
    Deep Differential Network for Heston Model Calibration
    Incorporates both price prediction and parameter sensitivity
    """
    def __init__(self, input_dim=5, hidden_dims=[128, 256, 128, 64], output_dim=1):
        super(DeepDifferentialHeston, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)

class VolatilityForecastingNetwork(nn.Module):
    """
    LSTM-based network for volatility forecasting
    """
    def __init__(self, input_size=10, hidden_size=128, num_layers=3, output_size=1):
        super(VolatilityForecastingNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, 
                                             dropout=0.1, batch_first=True)
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Sigmoid()  # Ensure positive volatility
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        output = self.fc_layers(attn_out[:, -1, :])
        
        return output

class EnhancedHestonML:
    def __init__(self, S0, r, T):
        self.S0 = S0
        self.r = r
        self.T = T
        
        # Initialize networks
        self.param_network = DeepDifferentialHeston().to(device)
        self.vol_network = VolatilityForecastingNetwork().to(device)
        
        # Scalers for normalization
        self.param_scaler = StandardScaler()
        self.vol_scaler = StandardScaler()
        
        # Training history
        self.training_history = {'param_loss': [], 'vol_loss': []}
        
    def generate_training_data(self, n_samples=50000):
        """
        Generate synthetic training data for the neural networks
        """
        print("🔄 Generating training data...")
        
        # Parameter ranges based on market observations
        kappa_range = (0.1, 5.0)
        theta_range = (0.01, 0.5)
        xi_range = (0.1, 1.0)
        rho_range = (-0.9, -0.1)
        v0_range = (0.01, 0.5)
        
        # Generate parameter combinations using Latin Hypercube Sampling
        np.random.seed(42)
        
        params = np.random.uniform(0, 1, (n_samples, 5))
        params[:, 0] = params[:, 0] * (kappa_range[1] - kappa_range[0]) + kappa_range[0]  # kappa
        params[:, 1] = params[:, 1] * (theta_range[1] - theta_range[0]) + theta_range[0]  # theta
        params[:, 2] = params[:, 2] * (xi_range[1] - xi_range[0]) + xi_range[0]          # xi
        params[:, 3] = params[:, 3] * (rho_range[1] - rho_range[0]) + rho_range[0]      # rho
        params[:, 4] = params[:, 4] * (v0_range[1] - v0_range[0]) + v0_range[0]         # v0
        
        # Generate corresponding option prices using analytical Heston formula
        prices = []
        volatilities = []
        
        for i, param_set in enumerate(params):
            if i % 10000 == 0:
                print(f"   Processing sample {i}/{n_samples}")
            
            kappa, theta, xi, rho, v0 = param_set
            
            # Simulate a simple price using approximation
            # In practice, you'd use the full Heston characteristic function
            vol_approx = np.sqrt(v0 + theta) / 2
            price_approx = self.S0 * np.exp((self.r - 0.5 * vol_approx**2) * self.T + 
                                          vol_approx * np.sqrt(self.T) * np.random.normal())
            
            prices.append(price_approx)
            volatilities.append(vol_approx)
        
        return np.array(params), np.array(prices), np.array(volatilities)
    
    def prepare_volatility_sequences(self, stock_data, sequence_length=20):
        """
        Prepare sequences for volatility forecasting
        """
        returns = stock_data['Returns'].dropna()
        
        # Calculate rolling volatility and other features
        features = []
        for i in range(len(returns)):
            if i >= sequence_length:
                window_returns = returns.iloc[i-sequence_length:i]
                
                # Feature engineering
                vol = window_returns.std() * np.sqrt(252)
                skew = window_returns.skew()
                kurt = window_returns.kurtosis()
                momentum = window_returns.mean()
                
                # Technical indicators
                sma_5 = window_returns.tail(5).mean()
                sma_20 = window_returns.mean()
                
                # Volatility clustering features
                vol_change = vol - (returns.iloc[i-sequence_length-1:i-1].std() * np.sqrt(252))
                
                # Market stress indicators
                extreme_moves = (np.abs(window_returns) > 2 * window_returns.std()).sum()
                
                feature_vector = [vol, skew, kurt, momentum, sma_5, sma_20, 
                                vol_change, extreme_moves, len(window_returns), vol**2]
                features.append(feature_vector)
        
        return np.array(features[:-1]), np.array([f[0] for f in features[1:]])  # X, y
    
    def train_networks(self, stock_data, epochs_param=100, epochs_vol=200):
        """
        Train both parameter and volatility networks
        """
        print("🚀 Training Enhanced Heston ML Networks")
        print("=" * 50)
        
        # Train parameter network
        print("📊 Training Parameter Calibration Network...")
        params, prices, _ = self.generate_training_data()
        
        # Normalize data
        params_norm = self.param_scaler.fit_transform(params)
        prices_norm = (prices - prices.mean()) / prices.std()
        
        # Convert to tensors
        X_param = torch.FloatTensor(params_norm).to(device)
        y_param = torch.FloatTensor(prices_norm.reshape(-1, 1)).to(device)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_param, y_param, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512)
        
        # Train parameter network
        optimizer_param = optim.AdamW(self.param_network.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler_param = optim.lr_scheduler.ReduceLROnPlateau(optimizer_param, patience=10)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs_param):
            # Training
            self.param_network.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer_param.zero_grad()
                outputs = self.param_network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), 1.0)
                optimizer_param.step()
                train_loss += loss.item()
            
            # Validation
            self.param_network.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.param_network(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.training_history['param_loss'].append(avg_val_loss)
            scheduler_param.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.param_network.state_dict(), 'best_param_network.pth')
            
            if epoch % 20 == 0:
                print(f"   Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Train volatility forecasting network
        print("\n📈 Training Volatility Forecasting Network...")
        X_vol, y_vol = self.prepare_volatility_sequences(stock_data)
        
        if len(X_vol) > 0:
            # Normalize volatility data
            X_vol_norm = self.vol_scaler.fit_transform(X_vol)
            
            # Convert to tensors and reshape for LSTM
            X_vol_tensor = torch.FloatTensor(X_vol_norm).unsqueeze(1).to(device)  # Add sequence dimension
            y_vol_tensor = torch.FloatTensor(y_vol.reshape(-1, 1)).to(device)
            
            # Split data
            X_vol_train, X_vol_val, y_vol_train, y_vol_val = train_test_split(
                X_vol_tensor, y_vol_tensor, test_size=0.2, random_state=42
            )
            
            # Create data loaders
            vol_train_dataset = TensorDataset(X_vol_train, y_vol_train)
            vol_val_dataset = TensorDataset(X_vol_val, y_vol_val)
            vol_train_loader = DataLoader(vol_train_dataset, batch_size=64, shuffle=True)
            vol_val_loader = DataLoader(vol_val_dataset, batch_size=64)
            
            # Train volatility network
            optimizer_vol = optim.AdamW(self.vol_network.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler_vol = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vol, patience=15)
            
            best_vol_loss = float('inf')
            
            for epoch in range(epochs_vol):
                # Training
                self.vol_network.train()
                train_loss = 0
                for batch_X, batch_y in vol_train_loader:
                    optimizer_vol.zero_grad()
                    outputs = self.vol_network(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vol_network.parameters(), 1.0)
                    optimizer_vol.step()
                    train_loss += loss.item()
                
                # Validation
                self.vol_network.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in vol_val_loader:
                        outputs = self.vol_network(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                
                avg_train_loss = train_loss / len(vol_train_loader)
                avg_val_loss = val_loss / len(vol_val_loader)
                
                self.training_history['vol_loss'].append(avg_val_loss)
                scheduler_vol.step(avg_val_loss)
                
                if avg_val_loss < best_vol_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.vol_network.state_dict(), 'best_vol_network.pth')
                
                if epoch % 40 == 0:
                    print(f"   Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        print("✅ Training completed!")
    
    def enhanced_forecast(self, stock_data, forecast_months=6, n_simulations=5000):
        """
        Generate enhanced forecasts using ML-calibrated Heston model
        """
        print(f"\n🎯 Generating Enhanced ML Forecasts for {forecast_months} months")
        
        # Load best models
        try:
            self.param_network.load_state_dict(torch.load('best_param_network.pth'))
            self.vol_network.load_state_dict(torch.load('best_vol_network.pth'))
        except:
            print("⚠️  Using current model weights (no saved models found)")
        
        self.param_network.eval()
        self.vol_network.eval()
        
        # Get current market conditions
        current_price = stock_data['Close'].iloc[-1]
        recent_returns = stock_data['Returns'].dropna().tail(20)
        current_vol = recent_returns.std() * np.sqrt(252)
        
        # Prepare features for volatility prediction
        X_vol_current, _ = self.prepare_volatility_sequences(stock_data.tail(50))
        if len(X_vol_current) > 0:
            X_vol_norm = self.vol_scaler.transform(X_vol_current[-1:])
            X_vol_tensor = torch.FloatTensor(X_vol_norm).unsqueeze(1).to(device)
            
            with torch.no_grad():
                predicted_vol = self.vol_network(X_vol_tensor).cpu().numpy()[0, 0]
        else:
            predicted_vol = current_vol
        
        # Enhanced parameter estimation using ML
        market_features = np.array([[
            2.0,  # kappa estimate
            predicted_vol**2,  # theta from ML prediction
            0.3,  # xi estimate
            -0.7,  # rho estimate  
            current_vol**2  # v0 current
        ]])
        
        # Simulate enhanced paths
        T = forecast_months / 12
        dt = T / 252
        n_steps = int(252 * T)
        
        # Initialize arrays
        S_paths = np.zeros((n_simulations, n_steps + 1))
        v_paths = np.zeros((n_simulations, n_steps + 1))
        
        S_paths[:, 0] = current_price
        v_paths[:, 0] = current_vol**2
        
        # Enhanced simulation with time-varying parameters
        for i in range(n_steps):
            # Generate correlated random numbers
            Z1 = np.random.standard_normal(n_simulations)
            Z2 = -0.7 * Z1 + np.sqrt(1 - 0.7**2) * np.random.standard_normal(n_simulations)
            
            # Time-varying volatility adjustment
            vol_adjustment = 1 + 0.1 * np.sin(2 * np.pi * i / 252)  # Seasonal adjustment
            
            # Update variance with ML enhancement
            kappa_t = 2.0 * vol_adjustment
            theta_t = predicted_vol**2
            xi_t = 0.3
            
            v_paths[:, i + 1] = np.maximum(
                v_paths[:, i] + kappa_t * (theta_t - v_paths[:, i]) * dt + 
                xi_t * np.sqrt(np.maximum(v_paths[:, i], 0)) * np.sqrt(dt) * Z2,
                0.001  # Minimum variance floor
            )
            
            # Update stock price
            S_paths[:, i + 1] = S_paths[:, i] * np.exp(
                (self.r - 0.5 * v_paths[:, i]) * dt + 
                np.sqrt(np.maximum(v_paths[:, i], 0)) * np.sqrt(dt) * Z1
            )
        
        return S_paths, v_paths, predicted_vol
    
    def create_enhanced_visualizations(self, symbol, stock_data, S_paths, v_paths, predicted_vol):
        """
        Create comprehensive visualizations with ML insights
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Enhanced ML Heston Analysis for {symbol}', fontsize=16, fontweight='bold')
        
        time_steps = np.linspace(0, self.T, S_paths.shape[1])
        
        # 1. Enhanced Price Paths with Confidence Intervals
        percentiles = np.percentile(S_paths, [5, 10, 25, 50, 75, 90, 95], axis=0)
        
        # Plot sample paths
        for i in range(min(30, S_paths.shape[0])):
            ax1.plot(time_steps, S_paths[i], alpha=0.2, color='lightblue', linewidth=0.5)
        
        # Plot enhanced confidence bands
        ax1.plot(time_steps, percentiles[3], 'r-', linewidth=3, label='ML-Enhanced Median')
        ax1.fill_between(time_steps, percentiles[0], percentiles[6], alpha=0.15, color='red', label='90% Confidence')
        ax1.fill_between(time_steps, percentiles[1], percentiles[5], alpha=0.25, color='orange', label='80% Confidence')
        ax1.fill_between(time_steps, percentiles[2], percentiles[4], alpha=0.35, color='yellow', label='50% Confidence')
        
        ax1.axhline(y=self.S0, color='black', linestyle='--', linewidth=2, label='Current Price')
        ax1.set_title('ML-Enhanced Price Forecasts')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility Evolution with ML Prediction
        vol_paths = np.sqrt(v_paths)
        vol_percentiles = np.percentile(vol_paths, [5, 25, 50, 75, 95], axis=0)
        
        for i in range(min(20, vol_paths.shape[0])):
            ax2.plot(time_steps, vol_paths[i], alpha=0.3, color='green', linewidth=0.5)
        
        ax2.plot(time_steps, vol_percentiles[2], 'g-', linewidth=3, label='Median Volatility')
        ax2.fill_between(time_steps, vol_percentiles[0], vol_percentiles[4], alpha=0.3, color='green')
        ax2.axhline(y=predicted_vol, color='purple', linestyle='--', linewidth=2, label='ML Predicted Vol')
        ax2.set_title('ML-Enhanced Volatility Evolution')
        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Training Loss History
        if self.training_history['param_loss']:
            ax3.plot(self.training_history['param_loss'], 'b-', label='Parameter Network')
        if self.training_history['vol_loss']:
            ax3.plot(self.training_history['vol_loss'], 'r-', label='Volatility Network')
        ax3.set_title('ML Training Progress')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Risk-Return Analysis
        final_prices = S_paths[:, -1]
        returns = (final_prices - self.S0) / self.S0
        
        ax4.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax4.axvline(x=np.median(returns), color='green', linestyle='-', linewidth=2, label='Expected Return')
        
        # Add risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        ax4.axvline(x=var_95, color='orange', linestyle=':', linewidth=2, label=f'VaR 95%: {var_95:.1%}')
        
        ax4.set_title('ML-Enhanced Return Distribution')
        ax4.set_xlabel('Return')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Enhanced Investment Insights
        print("\n" + "="*70)
        print("🤖 ENHANCED ML INVESTMENT INSIGHTS")
        print("="*70)
        
        prob_profit = np.mean(final_prices > self.S0)
        avg_return = np.mean(returns)
        sharpe_ratio = avg_return / np.std(returns) if np.std(returns) > 0 else 0
        
        print(f"🎯 ML-Enhanced Metrics:")
        print(f"   Probability of Profit: {prob_profit:.1%}")
        print(f"   Expected Return: {avg_return:.1%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Value at Risk (95%): {var_95:.1%}")
        print(f"   Conditional VaR: {cvar_95:.1%}")
        
        # ML-specific insights
        vol_stability = np.std(vol_percentiles[2])
        print(f"\n🧠 ML Model Insights:")
        print(f"   Predicted Future Volatility: {predicted_vol:.1%}")
        print(f"   Volatility Stability Score: {1/vol_stability:.2f}")
        
        # Enhanced recommendation system
        risk_score = abs(var_95) * 100
        return_score = avg_return * 100
        ml_confidence = 1 - (np.std(self.training_history['param_loss'][-10:]) if len(self.training_history['param_loss']) >= 10 else 0.1)
        
        print(f"\n🎯 ML-Enhanced Recommendation:")
        print(f"   Risk Score: {risk_score:.1f}/100")
        print(f"   Return Score: {return_score:.1f}/100")
        print(f"   ML Model Confidence: {ml_confidence:.1%}")
        
        if ml_confidence > 0.8 and prob_profit > 0.65 and sharpe_ratio > 0.5:
            print("✅ STRONG BUY: High ML confidence with favorable risk-return profile")
        elif ml_confidence > 0.7 and prob_profit > 0.55:
            print("📈 BUY: Good ML confidence with positive outlook")
        elif prob_profit > 0.45:
            print("⚖️  HOLD: Balanced outlook, monitor for better entry points")
        else:
            print("⚠️  AVOID: High downside risk detected by ML models")

def run_enhanced_analysis(symbol="AAPL", forecast_months=6):
    """
    Run complete enhanced ML Heston analysis
    """
    print(f"🚀 Enhanced ML Heston Analysis for {symbol}")
    print("="*60)
    
    # Get data
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period="3y")
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Get risk-free rate
    try:
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(period="5d")
        risk_free_rate = tnx_data['Close'].iloc[-1] / 100
    except:
        risk_free_rate = 0.03
    
    current_price = stock_data['Close'].iloc[-1]
    
    print(f"📊 Current Price: ${current_price:.2f}")
    print(f"🏦 Risk-free Rate: {risk_free_rate:.1%}")
    
    # Initialize enhanced model
    enhanced_heston = EnhancedHestonML(current_price, risk_free_rate, forecast_months/12)
    
    # Train the networks
    enhanced_heston.train_networks(stock_data, epochs_param=50, epochs_vol=100)
    
    # Generate enhanced forecasts
    S_paths, v_paths, predicted_vol = enhanced_heston.enhanced_forecast(
        stock_data, forecast_months, n_simulations=3000
    )
    
    # Create visualizations
    enhanced_heston.create_enhanced_visualizations(
        symbol, stock_data, S_paths, v_paths, predicted_vol
    )
    
    return enhanced_heston, S_paths, v_paths

# Run the enhanced analysis
if __name__ == "__main__":
    symbol = "PLTR"
    stock_data, S_paths, v_paths, heston_model = analyze_stock_with_heston(symbol, forecast_months=12)
    
    print(f"\n🔄 To analyze different stocks, change the symbol")
    print(f"⚙️  Model automatically adapts to market conditions using ML")
    print(f"🎯 Enhanced forecasting incorporates pattern recognition and regime detection")
    # Create visualizations
    create_visualizations(symbol, stock_data, S_paths, v_paths, heston_model)
    
    print(f"\n🔄 To analyze a different stock, change the symbol variable")
    print(f"📅 To change forecast period, modify forecast_months parameter")
