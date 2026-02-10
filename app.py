"""
Pair Trading Dashboard - Vercel Compatible Version
This is the main entry point for Vercel deployment
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import os

# -----------------------------
# PARAMETRAR - USER'S CUSTOM SETTINGS
# -----------------------------
WINDOW = 65

# AGGRESSIVE STOP LOSSES - Exit snabbt vid förluster
ENTRY_Z = -1.7         # Köp när undervärderad
STOP_LOSS_Z = -10     # TIGHTARE stop loss (var -3.0)
TRAILING_STOP_PCT = 0.5  # 50% trailing stop från högsta värde
STOP_LOSS_PCT = 0.18    # Hard stop at -18%

# PROFIT TARGETS - Låt vinnare springa
PARTIAL_EXIT_Z = 0.1   # Ta hem 50% av position här
FULL_EXIT_Z = 0.9       # Exit resterande position här
MIN_PROFIT_TO_HOLD = 0.05  # Minst 5% vinst innan vi håller för mer

# VOLATILITET FILTER
MAX_VOLATILITY = 0.2       # Undvik aktier med >20% daglig volatilitet
MIN_VOLUME_RATIO = 0.5  # Kräv minst 50% av genomsnittlig volym

# SEKTOR TREND FILTER (NEW)
SECTOR_TREND_DAYS = 5      # Kolla sektor-trend över senaste 5 dagarna
SECTOR_TREND_THRESHOLD = -0.02  # Skip om sektor ned mer än 2% över period

# ROBUST FEATURES
MAX_DAYS_NO_DATA = 3    # Emergency exit after 3 days without price data
MAX_DAYS_IN_TRADE = 90  # Force exit after 90 days

START_CAPITAL = 100000
MAX_POSITION_SIZE = 0.25  # 25% per position
TRANSACTION_COST = 0.001
SLIPPAGE = 0.000

dark_bg = "#0e0e0e"
text_color = "#e0e0e0"
accent = "#00ff7f"
danger = "#ff4d4d"
warning = "#ffa500"

# Swedish stocks (Stockholm Exchange)
swedish_stocks = {
    "ABB.ST": ["Industrials"],
    "ADDT-B.ST": ["Industrials"],
    "ALFA.ST": ["Industrials"],
    "ASSA-B.ST": ["Industrials"],
    "AZN.ST": ["Health Care"],
    "ATCO-A.ST": ["Industrials"],
    "BOL.ST": ["Materials"],
    "EPI-A.ST": ["Industrials"],
    "EQT.ST": ["Financials"],
    "ERIC-B.ST": ["Information Technology"],
    "ESSITY-B.ST": ["Consumer Staples"],
    "EVO.ST": ["Consumer Discretionary"],
    "SHB-A.ST": ["Financials"],
    "HM-B.ST": ["Consumer Discretionary"],
    "HEXA-B.ST": ["Information Technology"],
    "INDU-C.ST": ["Financials"],
    "INVE-B.ST": ["Financials"],
    "LIFCO-B.ST": ["Industrials"],
    "NIBE-B.ST": ["Industrials"],
    "NDA-SE.ST": ["Financials"],
    "SAAB-B.ST": ["Industrials"],
    "SAND.ST": ["Industrials"],
    "SCA-B.ST": ["Materials"],
    "SEB-A.ST": ["Financials"],
    "SKA-B.ST": ["Industrials"],
    "SKF-B.ST": ["Industrials"],
    "SWED-A.ST": ["Financials"],
    "TEL2-B.ST": ["Communication Services"],
    "TELIA.ST": ["Communication Services"],
    "VOLV-B.ST": ["Industrials"]
}

# Top 50 NASDAQ stocks (by market cap)
nasdaq_stocks = {
    # Mega cap tech
    "AAPL": ["Information Technology"],
    "MSFT": ["Information Technology"],
    "GOOGL": ["Communication Services"],
    "AMZN": ["Consumer Discretionary"],
    "NVDA": ["Information Technology"],
    "META": ["Communication Services"],
    "TSLA": ["Consumer Discretionary"],
    
    # Large cap tech
    "AVGO": ["Information Technology"],
    "ASML": ["Information Technology"],
    "COST": ["Consumer Staples"],
    "NFLX": ["Communication Services"],
    "AMD": ["Information Technology"],
    "ADBE": ["Information Technology"],
    "CSCO": ["Information Technology"],
    "PEP": ["Consumer Staples"],
    "TMUS": ["Communication Services"],
    "INTC": ["Information Technology"],
    "CMCSA": ["Communication Services"],
    "QCOM": ["Information Technology"],
    "TXN": ["Information Technology"],
    "INTU": ["Information Technology"],
    "AMGN": ["Health Care"],
    "HON": ["Industrials"],
    "AMAT": ["Information Technology"],
    "BKNG": ["Consumer Discretionary"],
    "SBUX": ["Consumer Discretionary"],
    "GILD": ["Health Care"],
    "ADP": ["Information Technology"],
    "MDLZ": ["Consumer Staples"],
    "REGN": ["Health Care"],
    "VRTX": ["Health Care"],
    "ISRG": ["Health Care"],
    "ADI": ["Information Technology"],
    "LRCX": ["Information Technology"],
    "PANW": ["Information Technology"],
    "MU": ["Information Technology"],
    "SNPS": ["Information Technology"],
    "PYPL": ["Financials"],
    "CDNS": ["Information Technology"],
    "KLAC": ["Information Technology"],
    "MRVL": ["Information Technology"],
    "CRWD": ["Information Technology"],
    "MELI": ["Consumer Discretionary"],
    "NXPI": ["Information Technology"],
    "FTNT": ["Information Technology"],
    "WDAY": ["Information Technology"],
    "DASH": ["Consumer Discretionary"],
    "ROST": ["Consumer Discretionary"],
    "AEP": ["Utilities"],
    "KDP": ["Consumer Staples"],
}

stocks = {**swedish_stocks, **nasdaq_stocks}

sectors = {}
for stock, inds in stocks.items():
    for ind in inds:
        if ind not in sectors:
            sectors[ind] = []
        sectors[ind].append(stock)

# Import the rest of your backtest code here
# For Vercel, we'll load pre-computed results from cache
# This avoids running the full backtest on every page load

# Initialize Dash app with proper configuration for Vercel
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # CRITICAL: Expose the Flask server for Vercel

# For now, let's create a simple loading screen
# In production, you'll want to pre-compute the backtest and load from cache

app.layout = html.Div(
    style={"backgroundColor": dark_bg, "color": text_color, "padding": "20px", "fontFamily": "Arial", "minHeight": "100vh"},
    children=[
        html.H2("🎯 Pair Trading Dashboard", style={"textAlign": "center"}),
        html.Div([
            html.P("Loading backtest results..."),
            html.P("Note: For Vercel deployment, backtest results should be pre-computed and cached.", 
                   style={"color": warning})
        ], style={"textAlign": "center", "marginTop": "50px"})
    ]
)

# The server object is what Vercel will use
if __name__ == "__main__":
    # Local development
    app.run(debug=True)