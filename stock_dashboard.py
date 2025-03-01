import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import re
import json
import ta  # Technical analysis library - pip install ta

# Initialize the app with a dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Stock Dashboard", className="text-center text-primary my-4"),
            dbc.InputGroup([
                dbc.Input(id="ticker-input", placeholder="Enter stock ticker (e.g., AAPL)", type="text", value="AAPL"),
                dbc.Button("Search", id="search-button", color="primary")
            ], className="mb-3")
        ], width=12)
    ]),
    
    # Main content area
    dbc.Row([
        # Left panel: Stock info and stats
        dbc.Col([
            html.Div(id="stock-header", className="mb-3"),
            
            html.Div(id="price-info", className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("Key Statistics"),
                dbc.CardBody(id="key-stats")
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("About"),
                dbc.CardBody(id="about-info")
            ], className="mb-3"),
        ], width=3),
        
        # Right panel: Chart and related info
        dbc.Col([
            # Chart period buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("1D", id="1d-button", outline=True, color="primary", className="me-1"),
                        dbc.Button("1W", id="1w-button", outline=True, color="primary", className="me-1"),
                        dbc.Button("1M", id="1m-button", outline=True, color="primary", className="me-1"),
                        dbc.Button("3M", id="3m-button", outline=True, color="primary", className="me-1"),
                        dbc.Button("1Y", id="1y-button", outline=True, color="primary", className="me-1"),
                        dbc.Button("5Y", id="5y-button", outline=True, color="primary")
                    ]),
                ], width=6),
                
                # Chart type toggle
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Candlestick", id="candlestick-view", outline=True, color="secondary", className="me-1"),
                        dbc.Button("Line", id="line-view", outline=True, color="secondary")
                    ]),
                ], width=6, className="text-end"),
            ], className="mb-2"),
            
            # Interval selection (only for 1D view)
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("1m", id="1min-interval", outline=True, color="secondary", className="me-1"),
                            dbc.Button("5m", id="5min-interval", outline=True, color="secondary")
                        ])
                    ], id="interval-buttons", style={"display": "none"}, className="mb-2"),
                ], width=6),
                
                # Technical Indicators Dropdown - NEW
                dbc.Col([
                    dbc.DropdownMenu(
                        label="Technical Indicators",
                        id="indicators-dropdown",
                        color="info",
                        children=[
                            # These will be dynamically updated based on timeframe
                            html.Div(id="indicators-menu-content", style={"minWidth": "300px", "padding": "10px"})
                        ],
                        direction="down",
                        className="ms-auto"
                    ),
                ], width=6, className="text-end"),
            ], className="mb-2"),
            
            # Stock chart
            dcc.Graph(id="price-chart", config={'displayModeBar': False}),
            
            # Information tabs
            dbc.Tabs([
                dbc.Tab([
                    html.Div(id="analyst-ratings", className="mt-3")
                ], label="Analyst Ratings"),
                dbc.Tab([
                    html.Div(id="earnings-data", className="mt-3")
                ], label="Earnings"),
            ], className="mt-3")
            
        ], width=9)
    ]),
    
    # Store the current timeframe for use in other callbacks
    dcc.Store(id="current-timeframe", data="1mo"),
], fluid=True, className="bg-dark text-light")

# Global variables for state management
current_period = "1mo"  # Default to 1-month view
current_interval = "1d"  # Default to 1-day intervals for 1-month
chart_type = "candlestick"  # Default to candlestick view

# Callback for the chart and associated data
@callback(
    [Output("price-chart", "figure"),
     Output("stock-header", "children"),
     Output("price-info", "children"),
     Output("key-stats", "children"),
     Output("about-info", "children"),
     Output("analyst-ratings", "children"),
     Output("earnings-data", "children"),
     Output("1d-button", "active"),
     Output("1w-button", "active"),
     Output("1m-button", "active"),
     Output("3m-button", "active"),
     Output("1y-button", "active"),
     Output("5y-button", "active"),
     Output("interval-buttons", "style"),
     Output("1min-interval", "active"),
     Output("5min-interval", "active"),
     Output("candlestick-view", "active"),
     Output("line-view", "active"),
     Output("current-timeframe", "data")],
    [Input("search-button", "n_clicks"),
     Input("1d-button", "n_clicks"),
     Input("1w-button", "n_clicks"),
     Input("1m-button", "n_clicks"),
     Input("3m-button", "n_clicks"),
     Input("1y-button", "n_clicks"),
     Input("5y-button", "n_clicks"),
     Input("1min-interval", "n_clicks"),
     Input("5min-interval", "n_clicks"),
     Input("candlestick-view", "n_clicks"),
     Input("line-view", "n_clicks")],
    [State("ticker-input", "value"),
     State("current-timeframe", "data"),
     # Get the state of all indicators using pattern-matching callback
     State({"type": "indicator-check", "index": dash.ALL}, "value"),
     State({"type": "indicator-check", "index": dash.ALL}, "id")]
)
def update_dashboard(search_click, d1_click, w1_click, m1_click, m3_click, y1_click, y5_click, 
                    min1_click, min5_click, candlestick_click, line_click, ticker,
                    current_timeframe, indicator_values, indicator_ids):
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "1m-button"  # Default to 1 month
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Map button to time period
    period_map = {
        "1d-button": "1d",
        "1w-button": "1wk",
        "1m-button": "1mo",
        "3m-button": "3mo",
        "1y-button": "1y",
        "5y-button": "5y",
        "search-button": "1mo",  # Default to 1-month when search is clicked
    }
    
    # Check if this is an interval button or chart view button click
    is_interval_button = button_id in ["1min-interval", "5min-interval"]
    is_chart_view_button = button_id in ["candlestick-view", "line-view"]
    
    # Keep track of current state for interval buttons and chart type using global variables
    global current_period, current_interval, chart_type
    
    # Update chart type if chart view button is clicked
    if is_chart_view_button:
        if button_id == "candlestick-view":
            chart_type = "candlestick"
        else:
            chart_type = "line"
    
    # Only change period if a period button was clicked
    if is_interval_button or is_chart_view_button:
        period = current_period
        # If we're not in 1D view but interval button is clicked,
        # force switch to 1D view
        if is_interval_button and period != "1d":
            period = "1d"
    else:
        # For period buttons, get period from the map
        period = period_map.get(button_id, "1mo")
    
    # Update current_period global
    current_period = period
    
    # Set interval based on period and button clicked
    if period == "1d":
        if button_id == "5min-interval":
            interval = "5m"
        elif button_id == "1min-interval":
            interval = "1m"
        else:
            # Maintain current interval if period button clicked
            interval = current_interval if current_interval in ["1m", "5m"] else "1m"
    elif period == "1wk":
        interval = "1h"  # Fixed for 1W
    elif period == "1mo":
        interval = "1d"  # Default for 1M - using 1-day interval
    elif period == "3mo":
        interval = "1d"  # Default for 3M
    else:
        interval = "1d"  # Default for longer periods
    
    # Update current_interval global
    current_interval = interval
    
    # Determine which period buttons are active
    period_button_states = [
        period == "1d",
        period == "1wk",
        period == "1mo",
        period == "3mo",
        period == "1y",
        period == "5y"
    ]
    
    # Show interval buttons only for 1D period
    interval_style = {"display": "block"} if period == "1d" else {"display": "none"}
    
    # Determine which interval buttons are active (only 1m and 5m for 1D view)
    interval_button_states = [
        interval == "1m",
        interval == "5m"
    ]
    
    # Determine which chart view buttons are active
    chart_type_states = [
        chart_type == "candlestick",
        chart_type == "line"
    ]
    
    try:
        # Ensure ticker is valid
        if not ticker or not re.match(r'^[A-Za-z\.\-]+$', ticker):
            ticker = "AAPL"  # Default to AAPL if invalid
        
        ticker = ticker.upper()  # Convert to uppercase
        
        # Get the stock data
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist_data = stock.history(period=period, interval=interval)
        
        if hist_data.empty:
            # Create empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title=f"No data available for {ticker}",
                template="plotly_dark",
                plot_bgcolor="#222",
                paper_bgcolor="#222",
                font=dict(color="white")
            )
            
            # Return empty data for all sections
            return (fig, 
                    f"No data for {ticker}", 
                    "No price information", 
                    "No statistics available", 
                    "No company information", 
                    "No analyst ratings", 
                    "No earnings data",
                    *period_button_states, interval_style, *interval_button_states, *chart_type_states, current_timeframe)
        
        # Create a dictionary to store which indicators are selected
        selected_indicators = {}
        for i, indicator_id in enumerate(indicator_ids):
            selected_indicators[indicator_id["index"]] = indicator_values[i]
        
        # Create the price chart
        fig = create_price_chart(hist_data, period, interval, chart_type, selected_indicators)
        
        # Get stock info for header
        info = stock.info
        
        # Create company header
        stock_header = create_stock_header(ticker, info)
        
        # Create price information section
        price_info = create_price_info(info, hist_data)
        
        # Create key statistics card
        key_stats = create_key_stats(info)
        
        # Create about section
        about_info = create_about_section(info)
        
        # Create analyst ratings
        analyst_ratings = create_analyst_ratings(info)
        
        # Create earnings data
        earnings_data = create_earnings_section(stock)
        
        return (fig, stock_header, price_info, key_stats, about_info, 
                analyst_ratings, earnings_data,
                *period_button_states, interval_style, *interval_button_states, *chart_type_states, current_timeframe)
    
    except Exception as e:
        # Create error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error loading data for {ticker}: {str(e)}",
            template="plotly_dark",
            plot_bgcolor="#222",
            paper_bgcolor="#222",
            font=dict(color="white")
        )
        
        # Return error message for all sections
        return (fig, 
                f"Error loading data for {ticker}", 
                f"Error: {str(e)}", 
                "Error loading statistics", 
                "Error loading company information", 
                "Error loading analyst ratings", 
                "Error loading earnings data",
                *period_button_states, interval_style, *interval_button_states,
                chart_type == "candlestick", chart_type == "line",
                current_timeframe)

# Updated create_price_chart function with technical indicators
def create_price_chart(hist_data, period, interval, chart_type, selected_indicators):
    try:
        # Calculate all the technical indicators based on what's selected
        if len(hist_data) >= 3:  # Need at least a few data points
            # Moving Averages
            if selected_indicators.get("sma-9", False):
                hist_data['SMA9'] = ta.trend.sma_indicator(hist_data['Close'], window=9)
            if selected_indicators.get("sma-20", False):
                hist_data['SMA20'] = ta.trend.sma_indicator(hist_data['Close'], window=20)
            if selected_indicators.get("sma-50", False):
                hist_data['SMA50'] = ta.trend.sma_indicator(hist_data['Close'], window=50)
            if selected_indicators.get("sma-100", False):
                hist_data['SMA100'] = ta.trend.sma_indicator(hist_data['Close'], window=100)
            if selected_indicators.get("sma-200", False):
                hist_data['SMA200'] = ta.trend.sma_indicator(hist_data['Close'], window=200)
                
            if selected_indicators.get("ema-9", False):
                hist_data['EMA9'] = ta.trend.ema_indicator(hist_data['Close'], window=9)
            if selected_indicators.get("ema-21", False):
                hist_data['EMA21'] = ta.trend.ema_indicator(hist_data['Close'], window=21)
            if selected_indicators.get("ema-55", False):
                hist_data['EMA55'] = ta.trend.ema_indicator(hist_data['Close'], window=55)
                
            if selected_indicators.get("vwap", False) and 'Volume' in hist_data.columns:
                # VWAP calculation - typically for intraday
                hist_data['VWAP'] = (hist_data['Close'] * hist_data['Volume']).cumsum() / hist_data['Volume'].cumsum()
            
            # Oscillators
            if selected_indicators.get("rsi-14", False):
                hist_data['RSI'] = ta.momentum.rsi(hist_data['Close'], window=14)
                
            if selected_indicators.get("macd-12-26-9", False):
                macd = ta.trend.MACD(hist_data['Close'], window_fast=12, window_slow=26, window_sign=9)
                hist_data['MACD'] = macd.macd()
                hist_data['MACD_Signal'] = macd.macd_signal()
                hist_data['MACD_Hist'] = macd.macd_diff()
                
            if selected_indicators.get("stochastic-14-3", False):
                stoch = ta.momentum.StochasticOscillator(hist_data['High'], hist_data['Low'], hist_data['Close'], window=14, smooth_window=3)
                hist_data['Stoch_K'] = stoch.stoch()
                hist_data['Stoch_D'] = stoch.stoch_signal()
            
            # Volatility
            if selected_indicators.get("bollinger-bands-20-2", False):
                bollinger = ta.volatility.BollingerBands(hist_data['Close'], window=20, window_dev=2)
                hist_data['BB_Upper'] = bollinger.bollinger_hband()
                hist_data['BB_Middle'] = bollinger.bollinger_mavg()
                hist_data['BB_Lower'] = bollinger.bollinger_lband()
                
            if selected_indicators.get("atr-14", False):
                hist_data['ATR'] = ta.volatility.average_true_range(hist_data['High'], hist_data['Low'], hist_data['Close'], window=14)
            
            # Volume
            if selected_indicators.get("volume-ma", False) and 'Volume' in hist_data.columns:
                hist_data['Volume_MA'] = ta.trend.sma_indicator(hist_data['Volume'], window=20)
        
        # Count how many separate plots we need
        separate_plots = 1  # Start with price chart
        if selected_indicators.get("rsi-14", False): separate_plots += 1
        if selected_indicators.get("macd-12-26-9", False): separate_plots += 1
        if selected_indicators.get("stochastic-14-3", False): separate_plots += 1
        if selected_indicators.get("atr-14", False): separate_plots += 1
        if selected_indicators.get("volume", False) or selected_indicators.get("volume-ma", False): separate_plots += 1
        
        # Create subplots - dynamically based on indicators
        row_heights = [0.5]  # Price chart gets 50% by default
        subplot_titles = ["Price"]
        
        # Add height and titles for each indicator subplot
        available_height = 0.5  # Remaining 50% to distribute
        if selected_indicators.get("rsi-14", False):
            row_heights.append(available_height / (separate_plots - 1))
            subplot_titles.append("RSI (14)")
        if selected_indicators.get("macd-12-26-9", False):
            row_heights.append(available_height / (separate_plots - 1))
            subplot_titles.append("MACD (12,26,9)")
        if selected_indicators.get("stochastic-14-3", False):
            row_heights.append(available_height / (separate_plots - 1))
            subplot_titles.append("Stochastic (14,3)")
        if selected_indicators.get("atr-14", False):
            row_heights.append(available_height / (separate_plots - 1))
            subplot_titles.append("ATR (14)")
        if selected_indicators.get("volume", False) or selected_indicators.get("volume-ma", False):
            row_heights.append(available_height / (separate_plots - 1))
            subplot_titles.append("Volume")
        
        # Create the subplots
        fig = make_subplots(rows=separate_plots, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.03, 
                          row_heights=row_heights,
                          subplot_titles=subplot_titles)
        
        # Add price data based on chart type
        if chart_type == "candlestick":
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name="Price",
                    increasing_line_color='#00C805',  # Green for up candles
                    decreasing_line_color='#FF5000',  # Red for down candles
                    line=dict(width=1)  # Only width is supported for candlestick lines
                ),
                row=1, col=1
            )
        else:
            # Line chart
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Close'],
                    mode='lines',
                    name="Price",
                    line=dict(color='#00C805', width=2)
                ),
                row=1, col=1
            )
        
        # Add moving averages to price chart
        ma_colors = {
            'SMA9': '#FF9900',    # Orange
            'SMA20': '#00FFFF',   # Cyan
            'SMA50': '#FF00FF',   # Magenta
            'SMA100': '#FFFF00',  # Yellow
            'SMA200': '#FFFFFF',  # White
            'EMA9': '#FF9900',    # Orange
            'EMA21': '#00FFFF',   # Cyan
            'EMA55': '#FF00FF',   # Magenta
            'VWAP': '#FFFF00'     # Yellow
        }
        
        for ma in ['SMA9', 'SMA20', 'SMA50', 'SMA100', 'SMA200', 'EMA9', 'EMA21', 'EMA55', 'VWAP']:
            if ma in hist_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data[ma],
                        mode='lines',
                        name=ma,
                        line=dict(color=ma_colors.get(ma, '#FFFFFF'), width=1),
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if 'BB_Upper' in hist_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dash'),
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255, 255, 255, 0.1)'
                ),
                row=1, col=1
            )
        
        # Track current row for adding indicators
        current_row = 2
        
        # Add RSI
        if 'RSI' in hist_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['RSI'],
                    mode='lines',
                    name="RSI",
                    line=dict(color='#FF9900', width=1),
                ),
                row=current_row, col=1
            )
            
            # Add reference lines
            fig.add_shape(type="line", x0=hist_data.index[0], x1=hist_data.index[-1], y0=70, y1=70,
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
                        row=current_row, col=1)
            fig.add_shape(type="line", x0=hist_data.index[0], x1=hist_data.index[-1], y0=30, y1=30,
                        line=dict(color="rgba(0, 255, 0, 0.5)", width=1, dash="dash"),
                        row=current_row, col=1)
            
            fig.update_yaxes(
                title_text="RSI",
                range=[0, 100],
                row=current_row, col=1
            )
            
            current_row += 1
        
        # Add MACD
        if 'MACD' in hist_data.columns:
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['MACD'],
                    mode='lines',
                    name="MACD",
                    line=dict(color='#00C805', width=1),
                    legendgroup="MACD"
                ),
                row=current_row, col=1
            )
            
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['MACD_Signal'],
                    mode='lines',
                    name="Signal",
                    line=dict(color='#FF9900', width=1),
                    legendgroup="MACD"
                ),
                row=current_row, col=1
            )
            
            # Histogram
            colors = ['#00C805' if val >= 0 else '#FF5000' for val in hist_data['MACD_Hist']]
            fig.add_trace(
                go.Bar(
                    x=hist_data.index,
                    y=hist_data['MACD_Hist'],
                    name="Histogram",
                    marker_color=colors,
                    legendgroup="MACD"
                ),
                row=current_row, col=1
            )
            
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            current_row += 1
        
        # Add Stochastic
        if 'Stoch_K' in hist_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Stoch_K'],
                    mode='lines',
                    name="%K",
                    line=dict(color='#00C805', width=1),
                    legendgroup="Stoch"
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Stoch_D'],
                    mode='lines',
                    name="%D",
                    line=dict(color='#FF9900', width=1),
                    legendgroup="Stoch"
                ),
                row=current_row, col=1
            )
            
            # Add reference lines
            fig.add_shape(type="line", x0=hist_data.index[0], x1=hist_data.index[-1], y0=80, y1=80,
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
                        row=current_row, col=1)
            fig.add_shape(type="line", x0=hist_data.index[0], x1=hist_data.index[-1], y0=20, y1=20,
                        line=dict(color="rgba(0, 255, 0, 0.5)", width=1, dash="dash"),
                        row=current_row, col=1)
            
            fig.update_yaxes(
                title_text="Stoch",
                range=[0, 100],
                row=current_row, col=1
            )
            
            current_row += 1
        
        # ATR
        if 'ATR' in hist_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data['ATR'],
                    mode='lines',
                    name="ATR",
                    line=dict(color='#FF9900', width=1),
                ),
                row=current_row, col=1
            )
            
            fig.update_yaxes(title_text="ATR", row=current_row, col=1)
            current_row += 1
        
        # Add volume if selected or if volume MA is selected
        if selected_indicators.get("volume", False) or 'Volume_MA' in hist_data.columns:
            colors = ['#FF5000' if row['Open'] > row['Close'] else '#00C805' for _, row in hist_data.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=hist_data.index,
                    y=hist_data['Volume'],
                    marker_color=colors,
                    name="Volume"
                ),
                row=current_row, col=1
            )
            
            # Add Volume MA if available
            if 'Volume_MA' in hist_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Volume_MA'],
                        mode='lines',
                        name="Volume MA",
                        line=dict(color='#FFFFFF', width=1),
                    ),
                    row=current_row, col=1
                )
        
        # Apply Robinhood-style formatting
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1E2023",
            paper_bgcolor="#1E2023",
            margin=dict(l=30, r=30, t=30, b=30),
            xaxis_rangeslider_visible=False,
            font=dict(color="#DCDCDC"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=650 if separate_plots > 1 else 500  # Make chart taller if we have indicators
        )
        
        # Format all y-axes - for price chart
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor="#333333",
            tickfont=dict(color="#DCDCDC"),
            tickprefix="$",
            autorange=True,
            row=1, col=1
        )
        
        # Format x-axis
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="#333333",
            tickfont=dict(color="#DCDCDC")
        )
        
        # Apply formatting based on period
        if period in ["1d", "1wk"]:
            # For 1-day and 1-week, include pre and post market hours
            fig.update_xaxes(
                tickformat="%H:%M" if period == "1d" else "%a %H:%M",  # Show day name for 1wk
                rangebreaks=[
                    # Hide only overnight hours (8:00 PM to 4:00 AM)
                    dict(bounds=["20:00", "04:00"], pattern="hour"),
                ]
            )
            
            # Add a shape to indicate regular trading hours (9:30 AM to 4:00 PM)
            for day in hist_data.index.floor('D').unique():
                # Add subtle shading for regular trading hours
                fig.add_shape(
                    type="rect",
                    x0=day.replace(hour=9, minute=30),
                    x1=day.replace(hour=16, minute=0),
                    y0=0,
                    y1=1,
                    yref="paper",
                    fillcolor="rgba(255, 255, 255, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add vertical lines for market open/close
                fig.add_shape(
                    type="line",
                    x0=day.replace(hour=9, minute=30),
                    x1=day.replace(hour=9, minute=30),
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash"),
                    layer="below"
                )
                fig.add_shape(
                    type="line",
                    x0=day.replace(hour=16, minute=0),
                    x1=day.replace(hour=16, minute=0),
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash"),
                    layer="below"
                )
        else:
            # Skip weekends for periods longer than 1 week
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"])  # hide weekends
                ]
            )
        
        return fig
        
    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        # Return empty figure on error
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#1E2023",
            paper_bgcolor="#1E2023",
            title=f"Error loading chart: {str(e)}"
        )
        return empty_fig

# Helper function to create stock header
def create_stock_header(ticker, info):
    try:
        company_name = info.get('longName', ticker)
        return html.Div([
            html.H2(company_name, className="mb-0"),
            html.H5(ticker, className="text-muted")
        ])
    except:
        return html.Div([
            html.H2(ticker, className="mb-0")
        ])

# Helper function to create price info section
def create_price_info(info, hist_data):
    try:
        current_price = info.get('currentPrice', hist_data['Close'].iloc[-1])
        previous_close = info.get('previousClose', hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price)
        
        price_change = current_price - previous_close
        price_change_percent = (price_change / previous_close) * 100
        
        change_color = "text-success" if price_change >= 0 else "text-danger"
        change_symbol = "+" if price_change >= 0 else ""
        
        return html.Div([
            html.H1(f"${current_price:.2f}", className="mb-0"),
            html.H5([
                f"{change_symbol}{price_change:.2f} ({change_symbol}{price_change_percent:.2f}%)"
            ], className=f"{change_color} mb-3")
        ])
    except:
        # Fallback if data is missing
        return html.Div([
            html.H1("$-.--", className="mb-0"),
            html.H5("-.-- (-.--%)", className="text-muted mb-3")
        ])

# Helper function to create key statistics
def create_key_stats(info):
    try:
        # Common financial metrics
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        eps = info.get('trailingEps', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
        volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 0)
        
        # Format market cap in billions/millions
        if market_cap >= 1_000_000_000:
            market_cap_formatted = f"${market_cap/1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            market_cap_formatted = f"${market_cap/1_000_000:.2f}M"
        else:
            market_cap_formatted = f"${market_cap:,.0f}"
        
        # Create the statistics table
        stats_table = html.Div([
            html.Table([
                html.Tr([
                    html.Td("Market Cap", className="text-muted"),
                    html.Td(market_cap_formatted, className="text-end")
                ]),
                html.Tr([
                    html.Td("P/E Ratio", className="text-muted"),
                    html.Td(f"{pe_ratio:.2f}" if pe_ratio else "N/A", className="text-end")
                ]),
                html.Tr([
                    html.Td("EPS (TTM)", className="text-muted"),
                    html.Td(f"${eps:.2f}" if eps else "N/A", className="text-end")
                ]),
                html.Tr([
                    html.Td("Dividend Yield", className="text-muted"),
                    html.Td(f"{dividend_yield:.2f}%" if dividend_yield else "N/A", className="text-end")
                ]),
                html.Tr([
                    html.Td("52 Week High", className="text-muted"),
                    html.Td(f"${fifty_two_week_high:.2f}" if fifty_two_week_high else "N/A", className="text-end")
                ]),
                html.Tr([
                    html.Td("52 Week Low", className="text-muted"),
                    html.Td(f"${fifty_two_week_low:.2f}" if fifty_two_week_low else "N/A", className="text-end")
                ]),
                html.Tr([
                    html.Td("Volume", className="text-muted"),
                    html.Td(f"{volume:,}" if volume else "N/A", className="text-end")
                ]),
                html.Tr([
                    html.Td("Avg. Volume", className="text-muted"),
                    html.Td(f"{avg_volume:,}" if avg_volume else "N/A", className="text-end")
                ])
            ], className="table table-sm table-dark")
        ])
        
        return stats_table
    except:
        return html.P("Statistics not available")

# Helper function to create about section
def create_about_section(info):
    try:
        business_summary = info.get('longBusinessSummary', 'No description available.')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        return html.Div([
            html.P(business_summary, className="mb-3"),
            html.Div([
                html.Span("Sector: ", className="text-muted"),
                html.Span(sector, className="me-3"),
                html.Span("Industry: ", className="text-muted"),
                html.Span(industry)
            ])
        ])
    except:
        return html.P("Company information not available")

# Helper function to create analyst ratings
def create_analyst_ratings(info):
    try:
        # Get analyst ratings
        recommendation = info.get('recommendationMean', 0)
        target_mean = info.get('targetMeanPrice', 0)
        target_high = info.get('targetHighPrice', 0)
        target_low = info.get('targetLowPrice', 0)
        
        # Recommendation scale: 1=Strong Buy, 3=Hold, 5=Strong Sell
        recommendation_text = "N/A"
        if recommendation:
            if recommendation <= 1.5:
                recommendation_text = "Strong Buy"
            elif recommendation <= 2.5:
                recommendation_text = "Buy"
            elif recommendation <= 3.5:
                recommendation_text = "Hold"
            elif recommendation <= 4.5:
                recommendation_text = "Sell"
            else:
                recommendation_text = "Strong Sell"
        
        # Create ratings visualization
        ratings_viz = html.Div([
            html.H4("Analyst Consensus:", className="mb-2"),
            html.H3(recommendation_text, className="mb-3"),
            
            html.Div([
                html.Div([
                    html.Span("Target Price Range:", className="text-muted"),
                    html.Div([
                        html.Span(f"${target_low:.2f}", className="text-danger"),
                        html.Span(" - ", className="mx-1"),
                        html.Span(f"${target_high:.2f}", className="text-success")
                    ], className="mt-1")
                ], className="me-4"),
                
                html.Div([
                    html.Span("Mean Target:", className="text-muted"),
                    html.Div(f"${target_mean:.2f}", className="mt-1")
                ])
            ], className="d-flex")
        ])
        
        return ratings_viz
    except:
        return html.P("Analyst ratings not available")

# Helper function to create earnings section
def create_earnings_section(stock):
    try:
        # Use income statement to get earnings data instead of deprecated earnings attribute
        income_stmt = stock.income_stmt
        
        if income_stmt is None or income_stmt.empty:
            return html.P("No earnings data available")
        
        # Extract relevant data - Net Income
        if 'Net Income' in income_stmt.index:
            net_income = income_stmt.loc['Net Income']
            
            # Format the data
            earnings_data = []
            for date, value in net_income.items():
                year = date.year
                earnings_data.append({"Year": year, "Net Income": value})
            
            # Sort by year
            earnings_data = sorted(earnings_data, key=lambda x: x["Year"])
            
            # Create earnings table
            rows = []
            for data in earnings_data:
                # Format the values for display
                net_income_fmt = f"${data['Net Income']/1000000:.2f}M" if data['Net Income'] else "N/A"
                
                rows.append(html.Tr([
                    html.Td(data["Year"]),
                    html.Td(net_income_fmt, className="text-end")
                ]))
            
            earnings_table = html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Year"),
                        html.Th("Net Income", className="text-end")
                    ])
                ),
                html.Tbody(rows)
            ], className="table table-sm table-dark")
            
            # Create a bar chart of earnings
            if len(earnings_data) > 1:
                years = [data["Year"] for data in earnings_data]
                values = [data["Net Income"] for data in earnings_data]
                
                colors = ["#00C805" if v >= 0 else "#FF5000" for v in values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=years,
                        y=values,
                        marker_color=colors
                    )
                ])
                
                fig.update_layout(
                    title="Net Income History",
                    xaxis_title="Year",
                    yaxis_title="Net Income ($)",
                    template="plotly_dark",
                    plot_bgcolor="#222",
                    paper_bgcolor="#222",
                    margin=dict(l=30, r=30, t=30, b=30),
                    font=dict(color="white"),
                    legend=dict(orientation="h"),
                    height=300
                )
                
                # Format y-axis to show in millions with autoscaling enabled
                fig.update_yaxes(
                    tickprefix="$",
                    ticksuffix="M",
                    # Remove the fixed tickvals and ticktext to allow autoscaling
                    tickformat=".1f",  # Format numbers with 1 decimal place
                    autorange=True     # Explicitly enable autoscaling
                )
                
                # Convert values to millions for display
                fig.update_traces(
                    y=[v/1000000 for v in values],
                    hovertemplate='Year: %{x}<br>Net Income: $%{y:.1f}M<extra></extra>'
                )
                
                earnings_chart = dcc.Graph(figure=fig, config={'displayModeBar': False})
                
                return html.Div([
                    earnings_chart,
                    earnings_table
                ])
            else:
                return earnings_table
        else:
            # If Net Income is not available in income statement
            return html.P("Net Income data not available")
    except Exception as e:
        print(f"Error in earnings section: {str(e)}")
        return html.P(f"Unable to load earnings data")

# Add a callback to populate the indicators menu based on timeframe
@callback(
    Output("indicators-menu-content", "children"),
    Input("current-timeframe", "data")
)
def update_indicator_menu(timeframe):
    # Define which indicators are relevant for each timeframe
    indicators_by_timeframe = {
        "1d": {
            "Moving Averages": ["SMA 9", "EMA 9", "VWAP"],
            "Oscillators": ["RSI (14)"],
            "Volatility": ["Bollinger Bands (20,2)"],
            "Volume": ["Volume", "Volume MA"]
        },
        "1wk": {
            "Moving Averages": ["SMA 9", "SMA 20", "EMA 9", "EMA 21"],
            "Oscillators": ["RSI (14)", "MACD (12,26,9)"],
            "Volatility": ["Bollinger Bands (20,2)", "ATR (14)"],
            "Volume": ["Volume", "Volume MA"]
        },
        "1mo": {
            "Moving Averages": ["SMA 9", "SMA 20", "SMA 50", "EMA 9", "EMA 21"],
            "Oscillators": ["RSI (14)", "MACD (12,26,9)", "Stochastic (14,3)"],
            "Volatility": ["Bollinger Bands (20,2)", "ATR (14)"],
            "Volume": ["Volume", "Volume MA"]
        },
        "3mo": {
            "Moving Averages": ["SMA 20", "SMA 50", "SMA 100", "EMA 21", "EMA 55"],
            "Oscillators": ["RSI (14)", "MACD (12,26,9)", "Stochastic (14,3)"],
            "Volatility": ["Bollinger Bands (20,2)", "ATR (14)"],
            "Volume": ["Volume", "Volume MA"]
        },
        "1y": {
            "Moving Averages": ["SMA 20", "SMA 50", "SMA 200", "EMA 21", "EMA 55"],
            "Oscillators": ["RSI (14)", "MACD (12,26,9)", "Stochastic (14,3)"],
            "Volatility": ["Bollinger Bands (20,2)", "ATR (14)"],
            "Volume": ["Volume", "Volume MA"]
        },
        "5y": {
            "Moving Averages": ["SMA 50", "SMA 100", "SMA 200", "EMA 55"],
            "Oscillators": ["RSI (14)", "MACD (12,26,9)"],
            "Volatility": ["Bollinger Bands (20,2)", "ATR (14)"],
            "Volume": ["Volume", "Volume MA"]
        }
    }
    
    # Default to 1mo if timeframe not found
    if timeframe not in indicators_by_timeframe:
        timeframe = "1mo"
    
    # Create the indicators menu content
    menu_content = []
    
    # Create a section for each indicator group
    for group, indicators in indicators_by_timeframe[timeframe].items():
        menu_content.append(html.H6(group, className="mt-2 text-info"))
        
        # Create checkboxes for each indicator
        checkbox_rows = []
        for i in range(0, len(indicators), 2):
            row_items = []
            # Add first indicator in row
            row_items.append(
                dbc.Col([
                    dbc.Checkbox(
                        id={"type": "indicator-check", "index": indicators[i].lower().replace(" ", "-").replace("(", "").replace(")", "").replace(",", "-")},
                        label=indicators[i],
                        value=False
                    ),
                ], width=6)
            )
            
            # Add second indicator if it exists
            if i + 1 < len(indicators):
                row_items.append(
                    dbc.Col([
                        dbc.Checkbox(
                            id={"type": "indicator-check", "index": indicators[i+1].lower().replace(" ", "-").replace("(", "").replace(")", "").replace(",", "-")},
                            label=indicators[i+1],
                            value=False
                        ),
                    ], width=6)
                )
            
            checkbox_rows.append(dbc.Row(row_items, className="mb-1"))
        
        menu_content.extend(checkbox_rows)
        menu_content.append(html.Hr(className="my-2"))
    
    return menu_content

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
