#!/usr/bin/env python3
"""
Default Configuration for Bitcoin Trading Bot SMA - Testnet Version Light

This file contains all default configuration values for the trading system.
It centralizes API credentials, endpoints, trading parameters, and risk management settings.

Usage:
    from default_config import Config
    config = Config()
    
    # Access configuration values
    api_url = config.API_URL
    position_size = config.POSITION_SIZE_USD
"""

import os
from typing import Dict, List, Set
from hyperliquid.utils import constants


class Config:
    """
    Centralized configuration class for the trading system.
    
    This class contains all configuration parameters including:
    - API endpoints and credentials
    - Trading parameters
    - Risk management settings
    - System behavior settings
    """
    
    # =============================================================================
    # API CONFIGURATION
    # =============================================================================
    
    # Hyperliquid API Configuration
    API_URL = constants.MAINNET_API_URL  # MAINNET - REAL MONEY!
    MAINNET_API_URL = constants.MAINNET_API_URL  # Available for production use
    
    # Account Configuration (loaded from executer/config.json)
    CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "executer", "config.json")
    
    # Default account credentials (will be overridden by config.json)
    DEFAULT_SECRET_KEY = "0x49123456789"
    DEFAULT_ACCOUNT_ADDRESS = "0x0c123456789"
    
    # =============================================================================
    # TRADING PARAMETERS
    # =============================================================================
    
    # Position Management
    MAX_POSITIONS = 15  # Maximum number of concurrent positions
    POSITION_SIZE_USD = 15  # Size of each position in USD
    
    # Order Offsets (percentage from current price)
    LONG_OFFSET_PERCENT = 0.08  # How far below current price to place long orders
    SHORT_OFFSET_PERCENT = 0.08  # How far above current price to place short orders
    
    # =============================================================================
    # RISK MANAGEMENT SETTINGS
    # =============================================================================
    
    # Stop Loss Configuration
    STOP_LOSS_PERCENT = 4.0  # Stop loss at 4% loss
    
    # Trailing Stop Configuration
    TRAILING_STOP_PERCENT = 1.2  # Trailing stop at 1.2% below max profit
    TRAILING_STOP_MIN_PROFIT = 2.0  # Minimum profit before trailing stop activates
    
    # Take Profit Configuration
    PROFIT_TAKE_THRESHOLD = 6.0  # Take profit when reaching 6.0%
    PROFIT_TAKE_DROP = 1.5  # Take profit when profit drops to 1.5%
    
    # =============================================================================
    # TIMING CONFIGURATION
    # =============================================================================
    
    # Trading Cycle Timing
    TRADING_CYCLE_MINUTES = 10  # Run trading cycle every 10 minutes
    POSITION_CHECK_SECONDS = 3  # Check positions every 3 seconds
    ORDER_MAX_AGE_HOURS = 24  # Cancel orders older than 24 hours
    
    # Auto-refresh Settings
    INTERFACE_REFRESH_SECONDS = 5  # Auto-refresh interface every 5 seconds
    
    # =============================================================================
    # SYSTEM BEHAVIOR SETTINGS
    # =============================================================================
    
    # Debug and Logging
    DEBUG_MODE = True  # Enable debug logging
    MAX_LOG_ENTRIES = 50  # Maximum number of log entries to keep in memory
    
    # Scanner Configuration
    SCANNER_TOP_RESULTS = 10  # Number of top results to show from scanner
    SCANNER_DISPLAY_RESULTS = 5  # Number of results to display in interface
    
    # =============================================================================
    # FALLBACK COIN LIST
    # =============================================================================
    
    # Fallback list of coins if dynamic fetching fails
    FALLBACK_COINS: Set[str] = {
        'ETH', 'SUI', 'JUP', 'NEAR', 'ONDO', 'TIA', 'OP', 'FIL', 'WLD', 'PENGU',
        'BTC', 'SOL', 'AVAX', 'DOGE', 'LINK', 'MATIC', 'ADA', 'DOT', 'LTC', 'RENDER',
        'AAVE', 'ALGO', 'APT', 'ARB', 'ATOM', 'BCH', 'BFUSD', 'BNB', 'BNSOL', 'BONK',
        'ENA', 'ETC', 'FET', 'HBAR', 'ICP', 'INJ', 'NEXO', 'PEPE', 'POL', 'QNT',
        'SEI', 'SHIB', 'TAO', 'TON', 'TRUMP', 'TRX', 'UNI', 'BIO', 'VET',
        'WBETH', 'XLM', 'XRP'
    }
    
    # =============================================================================
    # SCANNER SCRIPT PATHS
    # =============================================================================
    
    # Scanner script locations
    LOW_SCANNER_SCRIPT = 'data/multi_coin_scanner_find_lows.py'
    HIGH_SCANNER_SCRIPT = 'data/multi_coin_tops_scanner.py'
    
    # Report file prefixes
    LOW_OPPORTUNITIES_PREFIX = 'crypto_opportunities_'
    HIGH_OPPORTUNITIES_PREFIX = 'crypto_daily_highs_'
    REPORTS_DIRECTORY = 'reports'
    
    # =============================================================================
    # INTERFACE CONFIGURATION
    # =============================================================================
    
    # Display formatting
    DISPLAY_WIDTH = 80  # Terminal display width
    PRICE_DECIMAL_PLACES = 6  # Decimal places for price display
    PERCENTAGE_DECIMAL_PLACES = 2  # Decimal places for percentage display
    
    # Tab names
    AVAILABLE_TABS = ['home', 'positions', 'orders', 'scanner']
    DEFAULT_TAB = 'home'
    
    # =============================================================================
    # MULTI-SIG CONFIGURATION
    # =============================================================================
    
    # Multi-signature wallet settings (if used)
    MULTI_SIG_ENABLED = False  # Set to True to enable multi-sig functionality
    
    # =============================================================================
    # ENVIRONMENT DETECTION
    # =============================================================================
    
    @classmethod
    def is_testnet(cls) -> bool:
        """Check if currently configured for testnet"""
        return cls.API_URL == constants.TESTNET_API_URL
    
    @classmethod
    def is_mainnet(cls) -> bool:
        """Check if currently configured for mainnet"""
        return cls.API_URL == constants.MAINNET_API_URL
    
    @classmethod
    def get_environment_name(cls) -> str:
        """Get human-readable environment name"""
        if cls.is_testnet():
            return "TESTNET"
        elif cls.is_mainnet():
            return "MAINNET"
        else:
            return "CUSTOM"
    
    # =============================================================================
    # CONFIGURATION VALIDATION
    # =============================================================================
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """
        Validate configuration settings and return list of warnings/errors.
        
        Returns:
            List of validation messages (empty if all valid)
        """
        warnings = []
        
        # Check if using testnet for safety
        if cls.is_mainnet():
            warnings.append("⚠️ WARNING: Using MAINNET - ensure this is intentional!")
        
        # Validate position size
        if cls.POSITION_SIZE_USD <= 0:
            warnings.append("❌ ERROR: POSITION_SIZE_USD must be positive")
        
        # Validate max positions
        if cls.MAX_POSITIONS <= 0:
            warnings.append("❌ ERROR: MAX_POSITIONS must be positive")
        
        # Validate risk management settings
        if cls.STOP_LOSS_PERCENT <= 0:
            warnings.append("❌ ERROR: STOP_LOSS_PERCENT must be positive")
        
        if cls.TRAILING_STOP_PERCENT <= 0:
            warnings.append("❌ ERROR: TRAILING_STOP_PERCENT must be positive")
        
        # Validate timing settings
        if cls.TRADING_CYCLE_MINUTES <= 0:
            warnings.append("❌ ERROR: TRADING_CYCLE_MINUTES must be positive")
        
        if cls.POSITION_CHECK_SECONDS <= 0:
            warnings.append("❌ ERROR: POSITION_CHECK_SECONDS must be positive")
        
        # Check config file exists
        if not os.path.exists(cls.CONFIG_FILE_PATH):
            warnings.append(f"⚠️ WARNING: Config file not found at {cls.CONFIG_FILE_PATH}")
        
        return warnings
    
    # =============================================================================
    # CONFIGURATION SUMMARY
    # =============================================================================
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration"""
        print("=" * 80)
        print("🔧 TRADING BOT CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"Environment: {cls.get_environment_name()}")
        print(f"API URL: {cls.API_URL}")
        print(f"Max Positions: {cls.MAX_POSITIONS}")
        print(f"Position Size: ${cls.POSITION_SIZE_USD} USD")
        print(f"Stop Loss: {cls.STOP_LOSS_PERCENT}%")
        print(f"Trailing Stop: {cls.TRAILING_STOP_PERCENT}%")
        print(f"Trading Cycle: {cls.TRADING_CYCLE_MINUTES} minutes")
        print(f"Position Check: {cls.POSITION_CHECK_SECONDS} seconds")
        print(f"Debug Mode: {cls.DEBUG_MODE}")
        print("=" * 80)
        
        # Show validation warnings
        warnings = cls.validate_config()
        if warnings:
            print("⚠️ CONFIGURATION WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")
            print("=" * 80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_config() -> Config:
    """Get configuration instance"""
    return Config()

def get_api_url() -> str:
    """Get current API URL"""
    return Config.API_URL

def get_trading_params() -> Dict:
    """Get trading parameters as dictionary"""
    return {
        'max_positions': Config.MAX_POSITIONS,
        'position_size_usd': Config.POSITION_SIZE_USD,
        'long_offset_percent': Config.LONG_OFFSET_PERCENT,
        'short_offset_percent': Config.SHORT_OFFSET_PERCENT
    }

def get_risk_params() -> Dict:
    """Get risk management parameters as dictionary"""
    return {
        'stop_loss_percent': Config.STOP_LOSS_PERCENT,
        'trailing_stop_percent': Config.TRAILING_STOP_PERCENT,
        'trailing_stop_min_profit': Config.TRAILING_STOP_MIN_PROFIT,
        'profit_take_threshold': Config.PROFIT_TAKE_THRESHOLD,
        'profit_take_drop': Config.PROFIT_TAKE_DROP
    }

def get_timing_params() -> Dict:
    """Get timing parameters as dictionary"""
    return {
        'trading_cycle_minutes': Config.TRADING_CYCLE_MINUTES,
        'position_check_seconds': Config.POSITION_CHECK_SECONDS,
        'order_max_age_hours': Config.ORDER_MAX_AGE_HOURS,
        'interface_refresh_seconds': Config.INTERFACE_REFRESH_SECONDS
    }


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    # Print configuration summary when run directly
    Config.print_config_summary()
    
    # Validate configuration
    warnings = Config.validate_config()
    if not warnings:
        print("✅ Configuration validation passed!")
    else:
        print("⚠️ Configuration validation found issues:")
        for warning in warnings:
            print(f"  {warning}")
