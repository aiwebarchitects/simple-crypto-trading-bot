# 🚀 Quick Setup Guide

## Prerequisites
- **MetaMask Wallet** (no funds required - fresh wallet is fine)
- **Python 3.7+** installed on your system

## Step 1: Get Your Hyperliquid Account

Create account with fee bonus: https://app.hyperliquid.xyz/join/BONUS500

### After you created your Account - Switch first to Testnet
1. Visit: https://app.hyperliquid-testnet.xyz/API
2. Create API credentials:
   - **Name**: `testnet_api`
   - Click **"Generate"** → This creates your wallet address
   - Click **"Authorize API Wallet"** → This shows your private key
   - **Days Valid**: 180

⚠️ **IMPORTANT**: Copy and save your private key immediately! It won't be shown again.

### For Mainnet (Advanced Users)
Create account with fee bonus: https://app.hyperliquid.xyz/join/BONUS500
- Get 4% reduced trading fees
- https://app.hyperliquid.xyz/API

## Step 2: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/aiwebarchitects/simple-crypto-trading-bot
cd simple-crypto-trading-bot

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Add Your Credentials

You need to add your credentials in **THREE** files:

### File 1: `executer/config.json`
```json
{
    "secret_key": "YOUR_PRIVATE_KEY_HERE",
    "account_address": "YOUR_WALLET_ADDRESS_HERE"
}
```

### File 2: `default_config.py`
```python
# API CONFIGURATION
SECRET_KEY = "YOUR_PRIVATE_KEY_HERE"
ACCOUNT_ADDRESS = "YOUR_WALLET_ADDRESS_HERE"
```

### File 3: `account_infos.py
```python
# API CONFIGURATION
user_state = info.user_state("YOUR_WALLET_ADDRESS_HERE")
```


## Step 4: Get Test Funds (Testnet Only)

Visit the faucet to get free USDC for testing:
https://app.hyperliquid-testnet.xyz/drip

## Step 5: Start Trading

```bash
python3 trading_panel.py
```
after the Start, you can see the System Status on top.
to Start the System: Use Number 5 on your keyboard.


That's it! Your bot is ready to trade. 🎉

---

⚠️ **Safety First**: Always start with testnet to learn the system before using real money.


# Mutli Crypto Trading Bot - Low/High Reversal Strategy

A comprehensive cryptocurrency trading system that combines automated backtesting with live trading capabilities using optimized parameters. The system implements a low/high reversal strategy with intelligent parameter optimization and real-time risk management.

## 🎯 Strategy Overview

This trading system implements a **Low/High Reversal Strategy** that:

### Long Positions (Buy Low)
- Identifies significant price lows using technical analysis
- Enters long positions by buying **0.05-0.30% above** significant lows (optimized per coin)
- Targets price reversals after major drops

### Short Positions (Sell High)
- Identifies daily high points in price action
- Enters short positions by selling **0.05-0.30% below** daily highs (optimized per coin)
- Profits from price corrections after peaks

### Intelligent Risk Management
- **Coin-Specific Parameters**: Uses backtested optimal parameters for each cryptocurrency
- **Dynamic Stop Loss**: 3-8% stop losses optimized per coin
- **Adaptive Trailing Stops**: 1-4% trailing stops based on backtest results
- **Automatic Parameter Updates**: Refreshes optimization every 24 hours

## 🏗️ System Architecture

### 1. **Automated Backtesting Engine** (`backtest/`)
- **Data Collection**: Fetches top cryptocurrencies and historical data
- **Parameter Optimization**: Tests thousands of parameter combinations
- **Performance Analysis**: Generates comprehensive performance metrics
- **Results Storage**: Saves optimized parameters in `results.json`

### 2. **Live Trading Panel** (`trading_panel.py`)
- **Backtest Integration**: Automatically loads and uses optimized parameters
- **Real-time Monitoring**: Live position tracking and risk management
- **Interactive Interface**: Terminal-based UI with multiple tabs
- **Intelligent Automation**: Auto-refreshes backtest data when stale

### 3. **Market Scanners** (`data/`)
- **Low Scanner**: Identifies significant price lows for long opportunities
- **High Scanner**: Detects daily highs for short opportunities
- **Real-time Analysis**: Continuous market monitoring

### 4. **Order Execution** (`executer/`)
- **Hyperliquid Integration**: Direct API connection for order placement
- **Position Management**: Automated opening and closing of positions
- **Risk Controls**: Built-in safety mechanisms and validation

## 🚀 Getting Started

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# 1. Run the trading panel (includes automatic backtesting)
python3 trading_panel.py

# 2. Or run standalone backtest
python3 backtest/multi_token_backtester_low_high_reversal.py
```

## 📊 Trading Panel Features

### 🏠 **Home Tab**
- **System Status**: Real-time trading system status
- **Account Overview**: Current account value and positions
- **Backtest Status**: Age of optimization data and top performers
- **Activity Logs**: Recent system activities and alerts

### 📊 **Positions Tab**
- **Active Positions**: Real-time P&L and position details
- **Optimized Parameters**: Shows coin-specific stop loss and trailing stop settings
- **Backtest Performance**: Historical performance data for each coin
- **Risk Management**: Live trailing stop levels and profit tracking

### 📋 **Orders Tab**
- **Open Orders**: All pending buy/sell orders
- **Order Details**: Size, price, and value information
- **Order Management**: Quick access to order controls

### 🔍 **Scanner Tab**
- **Low Opportunities**: Real-time long entry opportunities
- **High Opportunities**: Real-time short entry opportunities
- **Significance Scores**: AI-powered opportunity ranking
- **Market Analysis**: Current vs. optimal entry prices

## 🔄 Automatic Backtest Integration

### Smart Parameter Management
The system automatically:

1. **Checks Backtest Age**: On startup, verifies if optimization data is fresh
2. **Auto-Refresh**: If data is >24 hours old, runs fresh backtest automatically
3. **Load Optimized Parameters**: Uses coin-specific optimized settings
4. **Fallback Safety**: Uses default parameters if optimization unavailable

### Coin-Specific Optimization
Each cryptocurrency gets its own optimized parameters:
```json
{
  "BTC": {
    "long_offset_percent": 0.1,
    "short_offset_percent": 0.15,
    "stop_loss_percent": 3.0,
    "trailing_stop_percent": 1.5,
    "total_return_pct": 19.05,
    "win_rate": 100.0
  }
}
```

## 🔍 Backtesting Engine Details

### 1. Data Collection
- Fetches top cryptocurrencies by market cap from CoinGecko
- Downloads historical price data from Binance API
- Uses hourly candlestick data for accurate backtesting
- Implements intelligent caching to reduce API calls

### 2. Opportunity Detection

#### Significant Lows Algorithm
- Analyzes 40+ data points around each potential low
- Requires minimum 1% drop before the low
- Requires minimum 1% recovery after the low
- Calculates significance scores based on drop magnitude and recovery

#### Daily Highs Algorithm
- Groups data by trading days
- Identifies the highest price point of each day
- Filters for days with meaningful price ranges (>1%)
- Ranks opportunities by daily volatility

### 3. Parameter Optimization
Tests multiple parameter combinations:
- **Long Offset**: 0.05% to 0.30% (entry above lows)
- **Short Offset**: 0.05% to 0.30% (entry below highs)
- **Stop Loss**: 3% to 8% (maximum loss per trade)
- **Trailing Stop**: 1% to 4% (profit protection)

### 4. Performance Scoring
Optimizes based on:
- **Total Return** (40% weight)
- **Win Rate** (30% weight)
- **Profit Factor** (20% weight)
- **Low Drawdown** (10% weight)

## 📁 Project Structure

```
├── trading_panel.py                                 # Main trading interface
├── backtest/
│   ├── multi_token_backtester_low_high_reversal.py  # Backtesting engine
│   ├── results.json                                 # Optimized parameters (auto-generated)
│   └── logs/                                        # Backtest logs
├── data/
│   ├── multi_coin_scanner_find_lows.py             # Low opportunity scanner
│   └── multi_coin_tops_scanner.py                  # High opportunity scanner
├── executer/
│   ├── create_long_order.py                        # Long position execution
│   ├── create_short_order.py                       # Short position execution
│   ├── close_position.py                           # Position closing
│   └── config.json                                 # API configuration
├── data_cache/                                      # Cached market data
├── reports/                                         # Scanner reports
├── logs/                                            # System logs
├── default_config.py                               # System configuration
├── trading_scanner.py                              # Market scanning logic
├── fetch_available_coins.py                        # Coin discovery
├── requirements.txt                                 # Dependencies
└── README.md                                        # This file
```

## ⚙️ Configuration

### System Settings (`default_config.py`)
```python
# Trading Parameters
MAX_POSITIONS = 15                    # Maximum concurrent positions
POSITION_SIZE_USD = 15                # Size per position in USDC
TRADING_CYCLE_MINUTES = 30            # How often to scan for opportunities

# Risk Management (defaults, overridden by backtest results)
STOP_LOSS_PERCENT = 5.0               # Default stop loss
TRAILING_STOP_PERCENT = 1.0           # Default trailing stop
TRAILING_STOP_MIN_PROFIT = 2.0        # Minimum profit before trailing

# Backtest Settings
BACKTEST_REFRESH_HOURS = 24           # Auto-refresh optimization data
```

## 🎮 Using the Trading Panel

### Navigation
- **1-4**: Switch between tabs (Home, Positions, Orders, Scanner)
- **5**: Start the trading system
- **6**: Stop the trading system
- **7**: Refresh display
- **8**: Exit application
- **9**: Manually close a position

### System Operation
1. **Start**: Launch `python3 trading_panel.py`
2. **Auto-Setup**: System automatically checks and refreshes backtest data
3. **Monitor**: Use tabs to monitor positions and opportunities
4. **Trade**: System automatically manages positions using optimized parameters

## 📈 Performance Monitoring

### Real-Time Metrics
- **Position P&L**: Live profit/loss tracking
- **Parameter Usage**: Shows optimized vs. default parameters
- **Backtest Performance**: Historical win rates and returns
- **Risk Levels**: Current stop loss and trailing stop levels

### Backtest Results Display
```
📊 BACKTEST STATUS:
Backtest Age: 2.3 hours
Coins Tested: 19 | Successful: 19
Optimized Parameters: 10 coins
Top Performers:
  1. SUI: 90.3% return, 100.0% win rate
  2. TRX: 53.6% return, 100.0% win rate
  3. XRP: 52.6% return, 100.0% win rate
```

## 🔧 Advanced Features

### Multi-Threading
- Concurrent backtesting of multiple cryptocurrencies
- Parallel market scanning and analysis
- Thread-safe logging and data handling

### Intelligent Caching
- 1-hour cache for market data
- Persistent backtest results storage
- Automatic cache management

### Error Handling
- Robust network error recovery
- Graceful degradation when data unavailable
- Comprehensive logging for debugging

### Safety Features
- Position size limits
- Maximum position count enforcement
- API rate limiting protection
- Automatic system shutdown on critical errors

## 📊 Interpreting Results

### Key Metrics
- **Total Return %**: Overall profit/loss from the strategy
- **Win Rate %**: Percentage of trades that were profitable
- **Profit Factor**: Gross profits ÷ Gross losses (>1.0 is profitable)
- **Max Drawdown %**: Largest decline from peak equity
- **Parameter Source**: 📈 Optimized vs ⚙️ Default parameters

### Position Display Example
```
BTC      LONG  5x   100.0        $43250     $43500     🟢2.89%   $12.50
         📈 Optimized | SL: 3.0% | TS: 1.5%
         Max Profit: 3.2% | Trailing Stop Level: 1.7%
         Backtest: 19.0% return, 100.0% win rate
```

## ⚠️ Risk Management

### Automated Controls
- **Stop Loss**: Automatic position closure on excessive losses
- **Trailing Stops**: Dynamic profit protection as positions move favorably
- **Position Limits**: Maximum number of concurrent positions
- **Size Controls**: Fixed position sizing to limit exposure

### Manual Controls
- **Emergency Stop**: Immediate system shutdown capability
- **Manual Position Closure**: Override automatic management
- **Parameter Monitoring**: Real-time visibility into risk settings

## 🚨 Important Disclaimers

### Trading Risks
- **Capital Loss**: Trading cryptocurrencies involves substantial risk
- **Market Volatility**: Crypto markets are highly volatile and unpredictable
- **Technical Risk**: System failures or bugs could result in losses
- **Execution Risk**: Network issues may affect order execution

### Backtesting Limitations
- **Historical Performance**: Past results don't guarantee future performance
- **Market Conditions**: Strategy performance varies with market conditions
- **Execution Assumptions**: Assumes perfect order execution at target prices
- **Transaction Costs**: May not fully account for fees and slippage

### Testnet vs Mainnet
- **Current Version**: Configured for Hyperliquid testnet
- **Real Trading**: Requires configuration changes for mainnet
- **Testing Recommended**: Thoroughly test before using real capital

## 🛠️ Development & Customization

### Adding New Features
- **Strategy Modifications**: Customize entry/exit logic
- **New Indicators**: Add technical analysis indicators
- **Risk Rules**: Implement additional risk management
- **UI Enhancements**: Extend the trading panel interface

### API Integration
- **Exchange Support**: Adapt for other exchanges
- **Data Sources**: Add alternative data providers
- **Notification Systems**: Integrate alerts and notifications

## 📞 Support & Contributing

For questions, issues, or contributions:
- Review the code documentation
- Check existing issues and logs
- Test thoroughly on testnet before mainnet use
- Follow proper risk management practices

---

**Trade Responsibly! 📊🚀**

*This system is for educational and research purposes. Always understand the risks and test thoroughly before trading with real capital.*
