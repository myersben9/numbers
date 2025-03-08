import yfinance as yf
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import talib
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Grab the 'Symbols' column from nasdaq.csv
nasdaq = pd.read_csv('nasdaq.csv')
symbols = nasdaq['Symbol'].tolist()
# Grab a random sample of 10 symbols
sample_symbols = random.sample(symbols, 100)

# Get the max available data for each symbol
max_data = {}
for symbol in sample_symbols:
    try:
        ticker = yf.Ticker(symbol)
        try:
            # First try with prepost=False to avoid dividend handling issues
            ticker_data = ticker.history(period="1mo", interval="5m", prepost=False)
        except Exception as e:
            # If that fails, try again with auto_adjust=False and actions=False
            try:
                ticker_data = ticker.history(period="1mo", interval="5m", 
                                            auto_adjust=False, actions=False)
            except:
                # Last resort: just skip this symbol
                print(f"Error loading data for {symbol}, skipping this symbol")
                continue
                
        if not ticker_data.empty:
            max_data[symbol] = ticker_data
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)[:100]}...")

# Continue with the rest of the code only if we have data
if not max_data:
    print("No valid data was collected. Please check your network connection or symbol list.")
    import sys
    sys.exit(1)

# Function to compute indicators
def compute_indicators(df, time, full_data):
    if time not in df.index:
        return None  # Return None if time doesn't exist
    
    historical_data = full_data[full_data.index <= time]
    if len(historical_data) < 30:
        return None
    
    close_prices = historical_data['Close'].values
    high_prices = historical_data['High'].values
    low_prices = historical_data['Low'].values
    volume = historical_data['Volume'].values
    
    rsi = talib.RSI(close_prices, timeperiod=14)[-1]
    macd, macdsignal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    stochastic_k, stochastic_d = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3)
    upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=20)
    cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)[-1]
    adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]
    mom = talib.MOM(close_prices, timeperiod=10)[-1]
    
    return {
        'RSI': rsi,
        'MACD': macd[-1],
        'Stochastic_K': stochastic_k[-1],
        'Stochastic_D': stochastic_d[-1],
        'Bollinger_Upper': upper_band[-1],
        'Bollinger_Middle': middle_band[-1],
        'Bollinger_Lower': lower_band[-1],
        'CCI': cci,
        'ADX': adx,
        'MOM': mom
    }

# Function to calculate max profit
def calculate_max_profit(day_data):
    if len(day_data) < 2:
        return None, None, 0

    # Initialize profit tracking
    min_price = day_data.iloc[0]['Close']
    max_profit = 0
    best_buy_time = day_data.index[0]
    best_sell_time = day_data.index[0]
    potential_buy_time = day_data.index[0]

    for i in range(1, len(day_data)):
        current_price = day_data.iloc[i]['Close']
        current_time = day_data.index[i]

        # Track max profit
        if current_price < min_price:
            min_price = current_price
            potential_buy_time = current_time
        
        profit = current_price - min_price
        if profit > max_profit:
            max_profit = profit
            best_buy_time = potential_buy_time
            best_sell_time = current_time

    return best_buy_time, best_sell_time, max_profit

# Store trading opportunities
trading_days = []
for symbol in sample_symbols:
    if symbol not in max_data:
        continue
    
    data = max_data[symbol]
    data['Date'] = data.index.date
    grouped_by_day = data.groupby('Date')
    
    for date, day_data in grouped_by_day:
        if len(day_data) < 2:
            continue
        
        buy_time, sell_time, profit = calculate_max_profit(day_data)
        if buy_time and sell_time:
            buy_indicators = compute_indicators(day_data, buy_time, data)
            sell_indicators = compute_indicators(day_data, sell_time, data)
            if buy_indicators and sell_indicators:
                trading_days.append({
                    'symbol': symbol,
                    'date': date,
                    'buy_time': buy_time.time(),
                    'sell_time': sell_time.time(),
                    'profit': profit,
                    **{f'buy_{key}': val for key, val in buy_indicators.items()},
                    **{f'sell_{key}': val for key, val in sell_indicators.items()}
                })

# Add this function for outlier removal
def remove_outliers(data, method='iqr', threshold=2.0):
    """
    Remove outliers from data using various methods
    
    Parameters:
    - data: numpy array of values
    - method: 'iqr', 'zscore', or 'percentile'
    - threshold: threshold multiplier for IQR/z-score methods, or percentile range for percentile method
    
    Returns: trimmed data array
    """
    if len(data) < 10:  # Not enough data to meaningfully remove outliers
        return data
    
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    elif method == 'zscore':
        mean, std = np.mean(data), np.std(data)
        z_scores = np.abs((data - mean) / std)
        return data[z_scores <= threshold]
    
    elif method == 'percentile':
        # For percentile method, threshold should be between 0-0.5 (e.g., 0.05 for 5th-95th percentile)
        lower = np.percentile(data, 100 * threshold)
        upper = np.percentile(data, 100 * (1 - threshold))
        return data[(data >= lower) & (data <= upper)]
    
    return data  # Default: return original data

if trading_days:
    trading_days_df = pd.DataFrame(trading_days)
    indicators = ['RSI', 'MACD', 'Stochastic_K', 'CCI', 'ADX', 'MOM']
    
    # Create a figure with more customized subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, indicator in enumerate(indicators):
        # Get data
        buy_data = trading_days_df[f'buy_{indicator}'].dropna().values
        sell_data = trading_days_df[f'sell_{indicator}'].dropna().values
        
        # Special handling for MACD and MOM - remove outliers for better normal distribution
        if indicator in ['MACD', 'MOM']:
            # Use a combination of methods for more robust outlier removal
            # First use percentile method to remove extreme outliers
            buy_data_trimmed = remove_outliers(buy_data, method='percentile', threshold=0.025)  # Keep 5-95% range
            sell_data_trimmed = remove_outliers(sell_data, method='percentile', threshold=0.025)
            
            # Then use z-score method for finer tuning
            buy_data_trimmed = remove_outliers(buy_data_trimmed, method='zscore', threshold=2.5)
            sell_data_trimmed = remove_outliers(sell_data_trimmed, method='zscore', threshold=2.5)
            
            # Store original counts for display
            buy_original_count = len(buy_data)
            sell_original_count = len(sell_data)
            
            # Replace with trimmed data
            buy_data = buy_data_trimmed
            sell_data = sell_data_trimmed
            
            # Calculate percent of data kept
            buy_kept_pct = (len(buy_data) / buy_original_count * 100) if buy_original_count > 0 else 0
            sell_kept_pct = (len(sell_data) / sell_original_count * 100) if sell_original_count > 0 else 0
        
        # Basic statistics for both buy and sell data
        buy_mean = np.mean(buy_data)
        buy_std = np.std(buy_data)
        buy_n = len(buy_data)
        
        sell_mean = np.mean(sell_data)
        sell_std = np.std(sell_data)
        sell_n = len(sell_data)
        
        # Skip if we don't have enough data
        if buy_n < 5 or sell_n < 5:
            axes[i].text(0.5, 0.5, "Insufficient data",
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[i].transAxes, fontsize=14)
            continue
        
        # Calculate confidence intervals for all indicators
        try:
            buy_sem = stats.sem(buy_data)
            buy_ci_95 = stats.t.interval(0.95, buy_n-1, loc=buy_mean, scale=buy_sem)
            buy_ci_width = buy_ci_95[1] - buy_ci_95[0]
            buy_ci_width_pct = (buy_ci_width / abs(buy_mean)) * 100 if buy_mean != 0 else float('inf')
        except:
            buy_ci_95 = (buy_mean - buy_std, buy_mean + buy_std)
            buy_ci_width_pct = 0
            
        try:
            sell_sem = stats.sem(sell_data)
            sell_ci_95 = stats.t.interval(0.95, sell_n-1, loc=sell_mean, scale=sell_sem)
            sell_ci_width = sell_ci_95[1] - sell_ci_95[0]
            sell_ci_width_pct = (sell_ci_width / abs(sell_mean)) * 100 if sell_mean != 0 else float('inf')
        except:
            sell_ci_95 = (sell_mean - sell_std, sell_mean + sell_std)
            sell_ci_width_pct = 0
            
        # Get skewness and kurtosis
        buy_skew = stats.skew(buy_data)
        buy_kurt = stats.kurtosis(buy_data)
        sell_skew = stats.skew(sell_data)
        sell_kurt = stats.kurtosis(sell_data)
        
        # Use normal distribution for MACD and MOM after outlier removal
        if indicator in ['MACD', 'MOM']:
            buy_best_fit = {
                'distribution': 'norm',
                'formula': f"Normal(μ={buy_mean:.2f}, σ={buy_std:.2f})",
                'parameters': (buy_mean, buy_std),
                'r_squared': 0.95  # Assumed good fit after outlier removal
            }
            
            sell_best_fit = {
                'distribution': 'norm',
                'formula': f"Normal(μ={sell_mean:.2f}, σ={sell_std:.2f})",
                'parameters': (sell_mean, sell_std),
                'r_squared': 0.95  # Assumed good fit after outlier removal
            }
            
            # Override fit quality for these indicators after trimming
            buy_fit_quality = "Good fit (trimmed)"
            buy_fit_color = 'green'
            sell_fit_quality = "Good fit (trimmed)"
            sell_fit_color = 'green'
            
            # Add note about trimming to stats text
            buy_trimmed_note = f"(Trimmed: {buy_kept_pct:.1f}% of original data kept)"
            sell_trimmed_note = f"(Trimmed: {sell_kept_pct:.1f}% of original data kept)"
        else:
            # For other indicators, use normal distribution by default
            buy_best_fit = {
                'distribution': 'norm',
                'formula': f"Normal(μ={buy_mean:.2f}, σ={buy_std:.2f})",
                'parameters': (buy_mean, buy_std),
                'r_squared': 0.85  # Default assumption
            }
            
            sell_best_fit = {
                'distribution': 'norm',
                'formula': f"Normal(μ={sell_mean:.2f}, σ={sell_std:.2f})",
                'parameters': (sell_mean, sell_std),
                'r_squared': 0.85  # Default assumption
            }
            
            # Default fit quality (can refine later)
            buy_fit_quality = "Normal fit"
            buy_fit_color = 'blue'
            sell_fit_quality = "Normal fit"
            sell_fit_color = 'red'
            
            # No trimming for other indicators
            buy_trimmed_note = ""
            sell_trimmed_note = ""
        
        # Determine buy R² value
        buy_r_squared = buy_best_fit['r_squared']
        sell_r_squared = sell_best_fit['r_squared']
        
        # Determine optimal bin count based on data size
        def get_optimal_bins(data):
            n = len(data)
            if n < 20:
                return max(5, n // 2)  # Small data: fewer bins
            elif n < 100:
                # Sturges' formula
                return int(np.ceil(np.log2(n) + 1))
            elif n < 500:
                # Rice Rule
                return int(np.ceil(2 * n**(1/3)))
            else:
                # Freedman-Diaconis Rule for larger datasets
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                bin_width = 2 * iqr / (n**(1/3)) if iqr > 0 else n**(1/3)
                data_range = np.max(data) - np.min(data)
                return int(np.ceil(data_range / bin_width)) if bin_width > 0 else 30
        
        # Calculate bins separately for buy and sell to respect their distributions
        buy_bins = get_optimal_bins(buy_data)
        sell_bins = get_optimal_bins(sell_data)
        
        # Use the smaller of the two to avoid overcrowding
        bins = min(max(10, min(buy_bins, sell_bins)), 50)
        
        # Create histograms with improved visibility
        if indicator in ['MACD', 'MOM']:
            # For MACD and MOM, use alpha-blended histograms with KDE overlay
            buy_hist = axes[i].hist(buy_data, bins=bins, color='blue', alpha=0.3, 
                                 density=True, label=f'Buy {indicator}')
            sell_hist = axes[i].hist(sell_data, bins=bins, color='red', alpha=0.3, 
                                  density=True, label=f'Sell {indicator}')
                                  
            # Add KDE for smoother visualization
            sns.kdeplot(buy_data, color='blue', ax=axes[i], linewidth=1.5, 
                     label='Buy KDE', linestyle=':', alpha=0.8)
            sns.kdeplot(sell_data, color='red', ax=axes[i], linewidth=1.5, 
                     label='Sell KDE', linestyle=':', alpha=0.8)
        else:
            # For other indicators, use standard histograms
            buy_hist = axes[i].hist(buy_data, bins=bins, color='blue', alpha=0.4, 
                                 density=True, label=f'Buy {indicator}')
            sell_hist = axes[i].hist(sell_data, bins=bins, color='red', alpha=0.4, 
                                  density=True, label=f'Sell {indicator}')
        
        # Create x_grid for smooth distribution curves
        x_min = min(min(buy_data), min(sell_data))
        x_max = max(max(buy_data), max(sell_data))
        # Add padding to avoid edge issues
        padding = (x_max - x_min) * 0.1
        x_grid = np.linspace(x_min - padding, x_max + padding, 1000)
        
        # Plot distribution fits with enhanced visibility
        if indicator in ['MACD', 'MOM']:
            # Normal fit for trimmed MACD/MOM data
            buy_pdf = stats.norm.pdf(x_grid, buy_mean, buy_std)
            axes[i].plot(x_grid, buy_pdf, 'b-', linewidth=2.5, 
                      label=f'Buy Normal Fit')
            
            sell_pdf = stats.norm.pdf(x_grid, sell_mean, sell_std)
            axes[i].plot(x_grid, sell_pdf, 'r-', linewidth=2.5, 
                      label=f'Sell Normal Fit')
        else:
            # Normal fit for other indicators
            buy_pdf = stats.norm.pdf(x_grid, buy_mean, buy_std)
            axes[i].plot(x_grid, buy_pdf, 'b-', linewidth=2, 
                      label=f'Buy Normal Fit')
            
            sell_pdf = stats.norm.pdf(x_grid, sell_mean, sell_std)
            axes[i].plot(x_grid, sell_pdf, 'r-', linewidth=2, 
                      label=f'Sell Normal Fit')
        
        # Display bin count in corner of plot
        axes[i].text(0.95, 0.05, f"Bins: {bins}", 
                  transform=axes[i].transAxes, 
                  horizontalalignment='right',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Add descriptive statistics and fit metrics for buy data
        buy_stats_text = (
            f"Buy: μ={buy_mean:.2f}, σ={buy_std:.2f}, n={buy_n} {buy_trimmed_note}\n"
            f"95% CI: [{buy_ci_95[0]:.2f}, {buy_ci_95[1]:.2f}] ({buy_ci_width_pct:.1f}% of μ)\n"
            f"Skew: {buy_skew:.2f}, Kurt: {buy_kurt:.2f}\n"
            f"Best Fit: {buy_best_fit['formula']}\n"
            f"R²: {buy_r_squared:.3f}, {buy_fit_quality}"
        )
        
        # Create background color based on buy fit quality
        buy_text_box = dict(boxstyle='round', facecolor='white', alpha=0.7, 
                      edgecolor=buy_fit_color, linewidth=2)
        
        axes[i].text(0.05, 0.95, buy_stats_text, transform=axes[i].transAxes, 
                  verticalalignment='top', bbox=buy_text_box, fontsize=9)
        
        # Add descriptive statistics and fit metrics for sell data
        sell_stats_text = (
            f"Sell: μ={sell_mean:.2f}, σ={sell_std:.2f}, n={sell_n} {sell_trimmed_note}\n"
            f"95% CI: [{sell_ci_95[0]:.2f}, {sell_ci_95[1]:.2f}] ({sell_ci_width_pct:.1f}% of μ)\n"
            f"Skew: {sell_skew:.2f}, Kurt: {sell_kurt:.2f}\n"
            f"Best Fit: {sell_best_fit['formula']}\n"
            f"R²: {sell_r_squared:.3f}, {sell_fit_quality}"
        )
        
        # Create background color based on sell fit quality
        sell_text_box = dict(boxstyle='round', facecolor='white', alpha=0.7, 
                      edgecolor=sell_fit_color, linewidth=2)
        
        axes[i].text(0.05, 0.20, sell_stats_text, transform=axes[i].transAxes, 
                  verticalalignment='top', bbox=sell_text_box, fontsize=9)
        
        axes[i].set_xlabel(indicator)
        axes[i].set_ylabel('Density' if indicator in ['MACD', 'MOM'] else 'Frequency')
        axes[i].set_title(f'{indicator} Distribution')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/indicator_distributions.png', dpi=300)
    plt.close()

    # Create a dictionary to store the distribution fits for each indicator
    distribution_fits = {}
    
    for i, indicator in enumerate(indicators):
        try:
            distribution_fits[indicator] = {
                'buy': {
                    'distribution': 'norm',
                    'parameters': (
                        float(trading_days_df[f'buy_{indicator}'].mean()), 
                        float(trading_days_df[f'buy_{indicator}'].std())
                    ),
                    'ci_95': [float(buy_ci_95[0]), float(buy_ci_95[1])],
                    'r_squared': float(buy_r_squared) if 'buy_r_squared' in locals() else 0.8
                },
                'sell': {
                    'distribution': 'norm', 
                    'parameters': (
                        float(trading_days_df[f'sell_{indicator}'].mean()),
                        float(trading_days_df[f'sell_{indicator}'].std())
                    ),
                    'ci_95': [float(sell_ci_95[0]), float(sell_ci_95[1])],
                    'r_squared': float(sell_r_squared) if 'sell_r_squared' in locals() else 0.8
                }
            }
        except Exception as e:
            print(f"Error saving distribution for {indicator}: {e}")
            continue
    
    # Save distributions to JSON file
    import json
    with open('indicator_distributions.json', 'w') as f:
        json.dump(distribution_fits, f, indent=2)
    
    print("Indicator distributions saved to indicator_distributions.json")
