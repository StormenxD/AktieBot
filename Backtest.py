from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
import numpy as np
from urllib.parse import parse_qs, urlparse

# PARAMETRAR
WINDOW = 65
ENTRY_Z = -1.7
STOP_LOSS_Z = -10
TRAILING_STOP_PCT = 0.5
STOP_LOSS_PCT = 0.18
PARTIAL_EXIT_Z = 0.1
FULL_EXIT_Z = 0.9
MIN_PROFIT_TO_HOLD = 0.05
MAX_VOLATILITY = 0.2
MIN_VOLUME_RATIO = 0.5
SECTOR_TREND_DAYS = 5
SECTOR_TREND_THRESHOLD = -0.02
MAX_DAYS_NO_DATA = 3
MAX_DAYS_IN_TRADE = 90
START_CAPITAL = 100000
MAX_POSITION_SIZE = 0.25
TRANSACTION_COST = 0.001
SLIPPAGE = 0.000

# Swedish stocks
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

# NASDAQ stocks
nasdaq_stocks = {
    "AAPL": ["Information Technology"],
    "MSFT": ["Information Technology"],
    "GOOGL": ["Communication Services"],
    "AMZN": ["Consumer Discretionary"],
    "NVDA": ["Information Technology"],
    "META": ["Communication Services"],
    "TSLA": ["Consumer Discretionary"],
    "AVGO": ["Information Technology"],
    "COST": ["Consumer Staples"],
    "NFLX": ["Communication Services"],
}

stocks = {**swedish_stocks, **nasdaq_stocks}

def calculate_zscore(series, window):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std

def run_backtest(selected_stocks):
    # Download data
    tickers = list(stocks.keys()) + ["^OMX", "^IXIC"]
    prices_raw = yf.download(
        tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
    
    # Download USD/SEK
    try:
        usdsek = yf.download("USDSEK=X", period="2y", interval="1d", auto_adjust=True, progress=False)["Close"]
        usdsek = usdsek.ffill(limit=5)
    except:
        usdsek = pd.Series(10.50, index=prices_raw.index)
    
    # Convert NASDAQ to SEK
    for ticker in nasdaq_stocks.keys():
        if ticker in prices_raw.columns:
            prices_raw[ticker] = prices_raw[ticker] * usdsek
    
    if "^IXIC" in prices_raw.columns:
        prices_raw["^IXIC"] = prices_raw["^IXIC"] * usdsek
    
    prices = prices_raw.ffill(limit=3)
    
    # Calculate signals
    signals = pd.DataFrame(index=prices.index)
    for stock in stocks.keys():
        if stock in prices.columns:
            signals[stock] = calculate_zscore(prices[stock], WINDOW)
    
    # Run trading simulation
    portfolio = START_CAPITAL
    cash = START_CAPITAL
    positions = {}
    portfolio_series = []
    trades = []
    
    for date in signals.index[WINDOW:]:
        portfolio_value = cash
        
        for stock in list(positions.keys()):
            if stock not in prices.columns:
                continue
                
            pos = positions[stock]
            current_price = prices.loc[date, stock]
            
            if pd.isna(current_price):
                continue
            
            position_value = pos['shares'] * current_price
            portfolio_value += position_value
            
            z = signals.loc[date, stock]
            profit_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            if z >= FULL_EXIT_Z:
                should_exit = True
                exit_reason = "Full Exit Z-score"
            elif profit_pct < -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = "Stop Loss"
            elif (date - pos['entry_date']).days >= MAX_DAYS_IN_TRADE:
                should_exit = True
                exit_reason = "Max Days"
            
            if should_exit:
                sell_value = pos['shares'] * current_price * (1 - TRANSACTION_COST - SLIPPAGE)
                cash += sell_value
                profit = sell_value - pos['entry_value']
                
                trades.append({
                    'date': date,
                    'stock': stock,
                    'action': 'SELL',
                    'shares': pos['shares'],
                    'price': current_price,
                    'value': sell_value,
                    'profit': profit,
                    'profit_pct': profit_pct * 100,
                    'reason': exit_reason
                })
                
                del positions[stock]
        
        # Entry signals
        for stock in selected_stocks:
            if stock not in signals.columns or stock in positions:
                continue
            
            z = signals.loc[date, stock]
            
            if z < ENTRY_Z and not pd.isna(z):
                position_size = min(cash * MAX_POSITION_SIZE, cash)
                current_price = prices.loc[date, stock]
                
                if pd.isna(current_price) or current_price <= 0:
                    continue
                
                shares = position_size / (current_price * (1 + TRANSACTION_COST + SLIPPAGE))
                entry_value = shares * current_price * (1 + TRANSACTION_COST + SLIPPAGE)
                
                positions[stock] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'entry_value': entry_value,
                    'entry_date': date,
                    'highest_value': entry_value
                }
                
                cash -= entry_value
                
                trades.append({
                    'date': date,
                    'stock': stock,
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'value': entry_value,
                    'profit': 0,
                    'profit_pct': 0,
                    'reason': 'Entry Signal'
                })
        
        portfolio_series.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': portfolio_value
        })
        portfolio = portfolio_value
    
    # Calculate benchmark returns
    benchmarks = {}
    if "^OMX" in prices.columns:
        omx_returns = (prices["^OMX"].pct_change()[WINDOW:] + 1).cumprod() * START_CAPITAL
        benchmarks['OMXS30'] = [{'date': d.strftime('%Y-%m-%d'), 'value': v} 
                                for d, v in zip(omx_returns.index, omx_returns.values)]
    
    if "^IXIC" in prices.columns:
        nasdaq_returns = (prices["^IXIC"].pct_change()[WINDOW:] + 1).cumprod() * START_CAPITAL
        benchmarks['NASDAQ'] = [{'date': d.strftime('%Y-%m-%d'), 'value': v} 
                                for d, v in zip(nasdaq_returns.index, nasdaq_returns.values)]
    
    return {
        'portfolio': portfolio_series,
        'benchmarks': benchmarks,
        'trades': trades,
        'final_value': portfolio,
        'total_return': ((portfolio - START_CAPITAL) / START_CAPITAL) * 100
    }

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query = parse_qs(parsed_path.query)
        
        selected_stocks = query.get('stocks', ['AAPL'])[0].split(',')
        
        try:
            result = run_backtest(selected_stocks)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        
        return