# Multi-Token Cryptocurrency Backtester

A comprehensive cryptocurrency trading backtester that uses Binance's free API to test MACD trading strategies on the top 100 coins by market cap.

## 🚀 What This Does

This tool automatically:
1. **Fetches the top 100 cryptocurrencies** by market cap from CoinGecko
2. **Matches them with Binance trading pairs** (USDT pairs only)
3. **Downloads historical price data** from Binance (4 days by default)
4. **Optimizes MACD parameters** for each coin using hundreds of combinations
5. **Simulates trading** with $100 starting capital
6. **Generates detailed plots** showing trading signals and performance
7. **Creates summary reports** comparing all coins
8. **finding the best 3**

## 📁 Files Included

- **`multi_token_backtester.py`** - Main backtester (no plots)
- **`multi_token_backtester_plot.py`** - Enhanced version with plotting
- **`README.md`** - This guide

## 🛠️ Requirements

### Python Dependencies
```bash
pip install pandas numpy requests matplotlib
```

### System Requirements
- Python 3.6 or higher
- Internet connection (for API calls)
- ~500MB free disk space (for data caching)

## 📊 How to Use

### Option 1: Basic Backtesting (No Plots)
```bash
python3 multi_token_backtester.py
```

### Option 2: Full Backtesting with Plots
```bash
python3 multi_token_backtester_plot.py
```

## 🔧 Configuration

You can modify these settings at the top of either script:

```python
BACKTEST_DAYS = 4              # Number of days to analyze
INITIAL_CAPITAL = 100.0        # Starting money ($100)
MAX_CONCURRENT_BACKTESTS = 3   # How many coins to process simultaneously

# MACD Optimization Ranges
OPT_SHORT_RANGE = (5, 25, 2)   # Short EMA: 5 to 25, step 2
OPT_LONG_RANGE = (20, 50, 3)   # Long EMA: 20 to 50, step 3
OPT_SIGNAL_RANGE = (4, 18, 2)  # Signal EMA: 4 to 18, step 2
```

## 📈 Understanding the Results

### Console Output
The tool will show real-time progress:
```
[BTC] Optimization Complete (679 combos in 7.33s). Best Params: {'short_ema': 21, 'long_ema': 44, 'signal_ema': 16}, Profit: -2.68%
[ETH] Optimization Complete (679 combos in 7.27s). Best Params: {'short_ema': 5, 'long_ema': 23, 'signal_ema': 4}, Profit: 1.80%
```

### Generated Files

#### 1. `tokens.json`
Contains the list of coins being analyzed:
```json
[
    {
        "symbol": "BTC",
        "name": "Bitcoin",
        "binance_pair": "BTCUSDT",
        "market_cap_rank": 1
    }
]
```

#### 2. `backtest_results.json`
Contains optimization results for each coin:
```json
{
    "BTC": {
        "short_ema": 21,
        "long_ema": 44,
        "signal_ema": 16,
        "last_run": "2025-08-26 15:25:22",
        "backtest_profit_pct": -2.68,
        "binance_pair": "BTCUSDT",
        "market_cap_rank": 1
    }
}
```

#### 3. Data Cache (`data_cache/` folder)
- Stores downloaded price data
- Files are cached for 1 hour to avoid re-downloading
- Format: `{SYMBOL}_{DAYS}d.csv`

#### 4. Plots (`plots/` folder) - Only with plot version
- **Individual coin plots**: `backtest_{SYMBOL}_{timestamp}.png`
- **Summary plot**: `summary_results_{timestamp}.png`

### Plot Explanations

#### Individual Coin Plots (3 panels):
1. **Top Panel**: Price chart with EMAs and buy/sell signals
   - Black line: Price
   - Blue line: Short EMA
   - Red line: Long EMA
   - Green triangles: Buy signals
   - Red triangles: Sell signals

2. **Middle Panel**: MACD indicator
   - Blue line: MACD line
   - Red line: Signal line
   - Green/Red areas: MACD histogram

3. **Bottom Panel**: Portfolio value over time
   - Green line: Your portfolio value
   - Gray dashed line: Starting capital ($100)

#### Summary Plot (2 panels):
1. **Top Panel**: Bar chart of all coins' performance
2. **Bottom Panel**: Scatter plot of profit vs market cap rank

## 🎯 Trading Strategy Explained

### MACD (Moving Average Convergence Divergence)
- **Short EMA**: Fast-moving average (typically 5-25 periods)
- **Long EMA**: Slow-moving average (typically 20-50 periods)
- **Signal EMA**: Smoothing of MACD line (typically 4-18 periods)

### Trading Signals
- **Buy Signal**: When MACD line crosses above signal line
- **Sell Signal**: When MACD line crosses below signal line

### Optimization Process
The tool tests hundreds of parameter combinations to find the most profitable settings for each coin over the specified time period.

## 📋 Typical Workflow

1. **First Run**: Tool fetches top 100 coins and creates `tokens.json`
2. **Data Download**: Historical price data is downloaded and cached
3. **Optimization**: Each coin is tested with 679 different parameter combinations
4. **Results**: Best parameters and profits are saved to `backtest_results.json`
5. **Plotting** (plot version): Visual charts are generated for analysis

## ⚠️ Important Notes

### This is for Educational/Research Purposes Only
- **Not financial advice**
- **Past performance doesn't predict future results**
- **Cryptocurrency trading is highly risky**

### Technical Limitations
- Uses only 4 days of data by default (short-term analysis)
- MACD is a lagging indicator
- No consideration for trading fees, slippage, or market conditions
- Assumes perfect order execution

### API Rate Limits
- **Binance**: 1200 requests per minute
- **CoinGecko**: 10-50 calls per minute (depending on plan)
- The tool includes delays to respect these limits

## 🔍 Troubleshooting

### Common Issues

#### "No module named 'matplotlib'"
```bash
pip install matplotlib
```

#### "HTTP Error 429" (Too Many Requests)
- Wait a few minutes and try again
- The tool has built-in rate limiting, but external factors can cause this

#### "No tokens found"
- Check your internet connection
- Verify that CoinGecko and Binance APIs are accessible

#### Empty plots or missing data
- Ensure matplotlib is installed
- Check that the `plots/` directory was created
- Verify sufficient disk space

### Performance Tips

1. **Reduce concurrent threads** if you have limited RAM:
   ```python
   MAX_CONCURRENT_BACKTESTS = 1
   ```

2. **Increase backtest period** for more reliable results:
   ```python
   BACKTEST_DAYS = 30
   ```

3. **Reduce optimization ranges** for faster execution:
   ```python
   OPT_SHORT_RANGE = (5, 15, 5)    # Fewer combinations
   OPT_LONG_RANGE = (20, 30, 5)
   OPT_SIGNAL_RANGE = (4, 10, 3)
   ```

## 📊 Sample Results Interpretation

### Good Performance Example
```
[ETH] Profit: 1.80%
```
- Strategy would have made 1.8% profit in 4 days
- $100 would become $101.80

### Poor Performance Example
```
[BTC] Profit: -2.68%
```
- Strategy would have lost 2.68% in 4 days
- $100 would become $97.32

### What to Look For
- **Consistent positive returns** across multiple coins
- **Low correlation with market cap rank** (strategy works on various coins)
- **Reasonable number of trades** (not over-trading)

## 🚀 Advanced Usage

### Custom Coin Lists
Edit `tokens.json` to test specific coins:
```json
[
    {
        "symbol": "BTC",
        "name": "Bitcoin",
        "binance_pair": "BTCUSDT",
        "market_cap_rank": 1
    },
    {
        "symbol": "ETH",
        "name": "Ethereum",
        "binance_pair": "ETHUSDT",
        "market_cap_rank": 2
    }
]
```

### Different Time Intervals
Modify the Binance API call in `fetch_historical_data()`:
```python
'interval': '4h',  # 4-hour candles instead of 1-hour
```

### Additional Indicators
The code structure allows easy addition of other technical indicators like RSI, Bollinger Bands, etc.

## 📞 Support

This is an educational tool. For issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure stable internet connection
4. Review the console output for specific error messages

## 📜 License

This project is for educational purposes. Use at your own risk.

---

**Remember**: This tool is for learning about algorithmic trading and technical analysis. Always do your own research and never invest more than you can afford to lose!
