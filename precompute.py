import yfinance as yf
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pickle
# -----------------------------
# PARAMETRAR - USER'S CUSTOM SETTINGS
# -----------------------------

WINDOW = 65
x = 0 # Put to 1 to enable reinvestment after loss, 0 to dissable

# AGGRESSIVE STOP LOSSES - Exit snabbt vid förluster
ENTRY_Z = -1.7         # Köp när undervärderad
STOP_LOSS_Z = -10     # TIGHTARE stop loss (var -3.0)
TRAILING_STOP_PCT = 0.5  # 50% trailing stop från högsta värde
STOP_LOSS_PCT = 0.18    # Hard stop at -18%
ReinvestPCT = 0.1
# PROFIT TARGETS - Låt vinnare springa
PARTIAL_EXIT_Z = 0.1   # Ta hem 50% av position här
FULL_EXIT_Z = 0.9       # Exit resterande position här
MIN_PROFIT_TO_HOLD = 0.05  # Minst 5% vinst innan vi håller för mer

# VOLATILITET FILTER
MAX_VOLATILITY = 0.2       # Undvik aktier med >20% daglig volatilitet
MIN_VOLUME_RATIO = 0.5  # Kräv minst 50% av genomsnittlig volym

# SEKTOR TREND FILTER
SECTOR_TREND_DAYS = 5      # Kolla sektor-trend över senaste 5 dagarna
SECTOR_TREND_THRESHOLD = -0.02  # Skip om sektor ned mer än 2% över period

# ROBUST FEATURES
MAX_DAYS_NO_DATA = 3  # Emergency exit after 3 days without price data
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

# -----------------------------
# AKTIER & SEKTORER
# -----------------------------

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
    "PEP": ["Consumer Staples"],
    "ADBE": ["Information Technology"],
    "CSCO": ["Information Technology"],
    "TMUS": ["Communication Services"],
    "CMCSA": ["Communication Services"],
    "INTC": ["Information Technology"],
    "INTU": ["Information Technology"],
    "QCOM": ["Information Technology"],
    "AMAT": ["Information Technology"],
    "TXN": ["Information Technology"],
    "AMGN": ["Health Care"],
    "HON": ["Industrials"],
    "BKNG": ["Consumer Discretionary"],
    "ISRG": ["Health Care"],
    "VRTX": ["Health Care"],
    "ADP": ["Information Technology"],
    "SBUX": ["Consumer Discretionary"],
    "GILD": ["Health Care"],
    "PANW": ["Information Technology"],
    "ADI": ["Information Technology"],
    "MU": ["Information Technology"],
    "LRCX": ["Information Technology"],
    "REGN": ["Health Care"],
    "MDLZ": ["Consumer Staples"],
    "PYPL": ["Financials"],
    "SNPS": ["Information Technology"],
    "KLAC": ["Information Technology"],
    "CDNS": ["Information Technology"],
    "MELI": ["Consumer Discretionary"],
    "MAR": ["Consumer Discretionary"],
    "CRWD": ["Information Technology"],
    "CTAS": ["Industrials"],
    "MRVL": ["Information Technology"],
    "ORLY": ["Consumer Discretionary"],
    "CSX": ["Industrials"],
    "ADSK": ["Information Technology"],
    "FTNT": ["Information Technology"],
    "DASH": ["Consumer Discretionary"],
    "ABNB": ["Consumer Discretionary"],
    "WDAY": ["Information Technology"],
    "NXPI": ["Information Technology"],
    "PCAR": ["Industrials"],
}

# Combine all stocks
stocks = {**swedish_stocks, **nasdaq_stocks}

print(f"\n📊 Total stocks loaded:")
print(f"  Swedish (OMXS30): {len(swedish_stocks)}")
print(f"  NASDAQ Top 50: {len(nasdaq_stocks)}")
print(f"  Total: {len(stocks)}")

sectors = {}
for stock, inds in stocks.items():
    for ind in inds:
        sectors.setdefault(ind, []).append(stock)

# -----------------------------
# HÄMTA DATA MED KVALITETSKONTROLL
# -----------------------------
print("\n" + "="*70)
print("Laddar ned data...")
print("="*70)
print("Detta kan ta 1-2 minuter med 80 aktier + valutakurs...")

# Download stock prices
prices_raw = yf.download(
    list(stocks.keys()) + ["^OMX", "^IXIC"],
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False
)["Close"]

# Download USD/SEK exchange rate
print("Laddar ner USD/SEK växelkurs...")
try:
    usdsek = yf.download(
        "USDSEK=X",
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]
    usdsek = usdsek.ffill(limit=5)  # Fill weekend gaps
    print(f"✓ USD/SEK kurs laddad (nuvarande: {usdsek.iloc[-1]:.2f})")
except Exception as e:
    print(f"⚠️ Kunde inte ladda USD/SEK kurs: {e}")
    print("   Använder fast kurs 10.50 SEK/USD")
    usdsek = pd.Series(10.50, index=prices_raw.index)

# Convert NASDAQ prices from USD to SEK
print("\nKonverterar NASDAQ-aktier från USD till SEK...")
nasdaq_tickers = list(nasdaq_stocks.keys())
converted_count = 0

for ticker in nasdaq_tickers:
    if ticker in prices_raw.columns:
        # Multiply USD price by USD/SEK rate to get SEK price
        prices_raw[ticker] = prices_raw[ticker] * usdsek
        converted_count += 1

print(f"✓ {converted_count} NASDAQ-aktier konverterade till SEK")
if 'AAPL' in prices_raw.columns and not prices_raw['AAPL'].isna().all():
    print(f"  Exempel: AAPL senaste pris: {prices_raw['AAPL'].iloc[-1]:.2f} SEK")

# Also convert NASDAQ Composite index to SEK for fair comparison
if "^IXIC" in prices_raw.columns:
    prices_raw["^IXIC"] = prices_raw["^IXIC"] * usdsek
    print(f"✓ NASDAQ Composite index konverterat till SEK")


# Check data quality BEFORE filling
print("\n📊 Datakvalitet:")
data_coverage = prices_raw.notna().sum() / len(prices_raw)
for stock in list(stocks.keys())[:84]:  # Show 84
    if stock in data_coverage:
        print(f"  {stock}: {data_coverage[stock]*100:.1f}% coverage")

# LIMITED forward-fill (max 3 days) - FIXED for new pandas
prices = prices_raw.ffill(limit=3)

# Identify days with large price gaps (potential data issues)
daily_returns = prices.pct_change()
large_gaps = (daily_returns.abs() > 0.25)  # 25%+ single-day moves

print("\n⚠️  Stora prisgap (>25% på en dag):")
gap_count = 0
for stock in list(stocks.keys()):
    if stock in large_gaps:
        gaps = large_gaps[stock].sum()
        if gaps > 0:
            print(f"  {stock}: {gaps} stora gap")
            gap_count += gaps

if gap_count == 0:
    print("  Inga stora gap hittades ✓")

returns = prices.pct_change()
volatility = returns.rolling(WINDOW).std()

# Hämta volym
try:
    volumes = yf.download(
        list(stocks.keys()),
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Volume"]
    volumes = volumes.ffill(limit=3).bfill(limit=3)
except:
    volumes = None
    print("⚠️  Kunde inte hämta volymdata")

sector_returns = {sector: returns[members].mean(axis=1) for sector, members in sectors.items()}
sector_df = pd.DataFrame(sector_returns)

signals = pd.DataFrame(index=returns.index)
for stock, inds in stocks.items():
    primary_sector = inds[0]
    diff = returns[stock] - sector_df[primary_sector]
    mean = diff.rolling(WINDOW).mean()
    std = diff.rolling(WINDOW).std()
    signals[stock] = (diff - mean) / std.replace(0, np.nan)

# -----------------------------
#BACKTEST MED DATA-HANTERING
# -----------------------------
def robust_backtest(signals_df, prices_df, prices_raw_df, volatility_df, volumes_df=None):
    """
    Robust backtest med:
    - Hantering av saknade data
    - Emergency exits
    - Debug logging för stop losses
    - Time-based exits
    """
    capital = START_CAPITAL
    cash = START_CAPITAL
    positions = {}
    
    portfolio_values = []
    buy_markers = {s: [] for s in signals_df.columns}
    sell_markers = {s: [] for s in signals_df.columns}
    partial_exit_markers = {s: [] for s in signals_df.columns}
    
    trades_log = []
    filtered_out = {'volatility': 0, 'volume': 0}
    emergency_exits = {'no_data': 0, 'time': 0}
    stop_loss_triggers = []  # Track when stops trigger
    
    for idx, date in enumerate(signals_df.index[WINDOW:], start=WINDOW):
        # Beräkna portföljvärde
        portfolio_value = cash
        for stock, pos in positions.items():
            # Use last valid price if current is NaN
            current_price = prices_df.loc[date, stock]
            if np.isnan(current_price) and 'last_valid_price' in pos:
                current_price = pos['last_valid_price']
            portfolio_value += pos['shares'] * current_price
        
        portfolio_values.append(portfolio_value)
        
        # === HANTERA BEFINTLIGA POSITIONER ===
        for stock in list(positions.keys()):
            z = signals_df.loc[date, stock]
            price = prices_df.loc[date, stock]
            price_raw = prices_raw_df.loc[date, stock]  # Check if real or forward-filled
            
            pos = positions[stock]
            days_held = (date - pos['entry_date']).days
            
            # === HANTERA SAKNADE DATA ===
            if np.isnan(price) or np.isnan(price_raw):
                # No real price data today
                pos['days_no_data'] = pos.get('days_no_data', 0) + 1
                
                # EMERGENCY EXIT: Too many days without data
                if pos['days_no_data'] >= MAX_DAYS_NO_DATA:
                    # Exit at last known valid price
                    if 'last_valid_price' in pos:
                        exit_price = pos['last_valid_price'] * (1 - SLIPPAGE)
                        profit_pct = (exit_price / pos['entry_price'] - 1) * 100
                        
                        proceeds = pos['shares'] * exit_price
                        cash += proceeds * (1 - TRANSACTION_COST)
                        
                        trades_log.append({
                            'date': date,
                            'stock': stock,
                            'action': 'SELL',
                            'reason': f'Emergency Exit (no data {pos["days_no_data"]} days)',
                            'profit_pct': profit_pct,
                            'entry_z': pos['entry_z'],
                            'exit_z': np.nan,
                            'days_held': days_held,
                            'estimated': True
                        })
                        
                        emergency_exits['no_data'] += 1
                        sell_markers[stock].append((date, pos.get('last_z', 0)))
                        del positions[stock]
                        print(f"⚠️  Emergency exit (no data): {stock} on {date.strftime('%Y-%m-%d')}, loss={profit_pct:.1f}%")
                continue  # Skip to next position
            else:
                # Valid price data - update tracking
                pos['last_valid_price'] = price
                pos['last_z'] = z
                pos['days_no_data'] = 0
            
            # === CALCULATE P&L WITH VALID DATA ===
            profit_pct = (price / pos['entry_price'] - 1)
            reinvested = pos['reinvested']

            # Update highest price for trailing stop
            if price > pos['highest_price']:
                pos['highest_price'] = price
            
            trailing_stop_price = pos['highest_price'] * (1 - TRAILING_STOP_PCT)
            
            should_exit = False
            exit_portion = 1.0
            exit_reason = ""
            
            # === EXIT CONDITIONS (IN PRIORITY ORDER) ===
            
            # 🚨 HARD PERCENTAGE STOP LOSS (HIGHEST PRIORITY)
            if profit_pct < -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = f"Hard Stop Loss ({profit_pct*100:.1f}%)"
                stop_loss_triggers.append({
                    'date': date,
                    'stock': stock,
                    'loss': profit_pct * 100,
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'days_held': days_held
                })
                print(f"🚨 HARD STOP TRIGGERED: {stock} on {date.strftime('%Y-%m-%d')} at {profit_pct*100:.1f}% loss")
           # === REINVEST / AVERAGE DOWN (ONCE ONLY) ===
            if (
                profit_pct <= -ReinvestPCT
                and not pos['reinvested']
                and x == 1
            ):
                    # Cap reinvest to a fraction of EXISTING position
                    current_position_value = pos['shares'] * price
                    reinvest_value = current_position_value * 1  # 50% add-on max

                    reinvest_value = min(reinvest_value, cash * 0.95)

                    if reinvest_value < 1000:
                        pass  # too small to matter
                    else:
                        buy_price = price * (1 + SLIPPAGE)
                        fee = reinvest_value * TRANSACTION_COST
                        new_shares = (reinvest_value - fee) / buy_price
                        cost = new_shares * buy_price + fee

                        if cost <= cash:
                            cash -= cost

                            # --- VWAP update (THIS IS CRITICAL) ---
                            total_shares = pos['shares'] + new_shares
                            total_cost = (
                                pos['shares'] * pos['entry_price']
                                + new_shares * buy_price
                            )

                            pos['shares'] = total_shares
                            pos['entry_price'] = total_cost / total_shares
                            pos['reinvested'] = True

                            # highest_price stays UNCHANGED
                            # entry_date stays UNCHANGED
                            # partial_exit stays UNCHANGED

                            buy_markers[stock].append((date, z))

                            trades_log.append({
                                'date': date,
                                'stock': stock,
                                'action': 'REINVEST',
                                'price': buy_price,
                                'shares': new_shares,
                                'z_score': z,
                            })



            # ⏰ TIME-BASED EXIT (prevent stale positions)
            elif days_held >= MAX_DAYS_IN_TRADE:
                should_exit = True
                exit_reason = f"Time Stop ({days_held} days)"
                emergency_exits['time'] += 1
            
            # 📉 TRAILING STOP
            elif price < trailing_stop_price and profit_pct > 0:
                should_exit = True
                exit_reason = f"Trailing Stop (från {pos['highest_price']:.2f})"
            
            # 📊 Z-SCORE STOP LOSS
            elif z < STOP_LOSS_Z:
                should_exit = True
                exit_reason = f"Z-Score Stop (z={z:.2f})"
            
            # 📉 NEGATIVE MOMENTUM
            elif profit_pct < -0.05 and z < pos['entry_z']:
                should_exit = True
                exit_reason = f"Negative Momentum"
            
            # 💰 PARTIAL EXIT
            elif z > PARTIAL_EXIT_Z and not pos.get('partial_exit', False) and profit_pct > MIN_PROFIT_TO_HOLD:
                should_exit = True
                exit_portion = 0.5
                exit_reason = f"Partial Exit (50%)"
            
            # ✅ FULL EXIT
            elif z > FULL_EXIT_Z:
                should_exit = True
                exit_reason = f"Full Exit (mean reversion)"
            
            # === EXECUTE EXIT ===
            if should_exit:
                shares_to_sell = pos['shares'] * exit_portion
                sell_price = price * (1 - SLIPPAGE)
                proceeds = shares_to_sell * sell_price
                transaction_fee = proceeds * TRANSACTION_COST
                cash += proceeds - transaction_fee
                
                final_profit_pct = (sell_price / pos['entry_price'] - 1) * 100
                
                trades_log.append({
                    'date': date,
                    'stock': stock,
                    'action': 'SELL',
                    'portion': exit_portion,
                    'reason': exit_reason,
                    'price': sell_price,
                    'shares': shares_to_sell,
                    'profit_pct': final_profit_pct,
                    'entry_z': pos['entry_z'],
                    'exit_z': z,
                    'days_held': days_held,
                    'entry_date': pos['entry_date']
                })
                
                if exit_portion == 0.5:
                    pos['shares'] *= 0.5
                    pos['partial_exit'] = True
                    partial_exit_markers[stock].append((date, z))
                else:
                    sell_markers[stock].append((date, z))
                    del positions[stock]
        
        # === SÖK NYA MÖJLIGHETER ===
        for stock in signals_df.columns:
            if stock in positions:
                continue
            
            z = signals_df.loc[date, stock]
            price = prices_df.loc[date, stock]
            price_raw = prices_raw_df.loc[date, stock]
            vol = volatility_df.loc[date, stock] if date in volatility_df.index else np.nan
            
            # Skip if no real data (only forward-filled)
            if np.isnan(price_raw):
                continue
            
            if np.isnan(z) or np.isnan(price) or price <= 0:
                continue
            
            # VOLATILITY FILTER
            if not np.isnan(vol) and vol > MAX_VOLATILITY:
                filtered_out['volatility'] += 1
                continue
            
            # VOLUME FILTER
            if volumes_df is not None and stock in volumes_df.columns:
                current_volume = volumes_df.loc[date, stock]
                avg_volume = volumes_df[stock].rolling(20).mean().loc[date]
                if not np.isnan(current_volume) and not np.isnan(avg_volume):
                    if current_volume < avg_volume * MIN_VOLUME_RATIO:
                        filtered_out['volume'] += 1
                        continue
            
            # ENTRY SIGNAL
            if z < ENTRY_Z:
                # 🛡️ SECTOR TREND FILTER - Avoid buying when entire sector is falling
                primary_sector = stocks[stock][0]
                
                # Calculate sector's trend over SECTOR_TREND_DAYS
                if len(sector_df[primary_sector]) >= SECTOR_TREND_DAYS:
                    sector_period_returns = sector_df[primary_sector].iloc[-SECTOR_TREND_DAYS:]
                    sector_cumulative = (1 + sector_period_returns).prod() - 1
                    
                    # Skip if sector down more than threshold
                    if sector_cumulative < SECTOR_TREND_THRESHOLD:
                        # Sector in downtrend - skip this trade (catching falling knife)
                        filtered_out['sector_downtrend'] = filtered_out.get('sector_downtrend', 0) + 1
                        continue
                
                position_value = portfolio_value * MAX_POSITION_SIZE
                
                if cash < position_value:
                    position_value = cash * 0.95
                
                if position_value < 1000:
                    continue
                
                buy_price = price * (1 + SLIPPAGE)
                transaction_fee = position_value * TRANSACTION_COST
                shares = (position_value - transaction_fee) / buy_price
                cost = shares * buy_price + transaction_fee
                reinvested = False
                if cost <= cash:
                    cash -= cost
                    positions[stock] = {
                        'shares': shares,
                        'entry_price': buy_price,
                        'entry_z': z,
                        'highest_price': buy_price,
                        'partial_exit': False,
                        'entry_date': date,
                        'last_valid_price': buy_price,
                        'last_z': z,
                        'days_no_data': 0,
                        'reinvested': False
                    }
                    
                    buy_markers[stock].append((date, z))
                    
                    trades_log.append({
                        'date': date,
                        'stock': stock,
                        'action': 'BUY',
                        'price': buy_price,
                        'shares': shares,
                        'z_score': z,
                        'volatility': vol
                    })
    
    portfolio_series = pd.Series(portfolio_values, index=signals_df.index[WINDOW:])
    trades_df = pd.DataFrame(trades_log)
    
    print(f"\n⚠️  Emergency Exits:")
    print(f"  No data: {emergency_exits['no_data']}")
    print(f"  Time limit: {emergency_exits['time']}")
    
    print(f"\n🛡️  Trade Filters:")
    print(f"  Filtrerade pga hög volatilitet: {filtered_out['volatility']}")
    print(f"  Filtrerade pga låg volym: {filtered_out['volume']}")
    print(f"  Filtrerade pga sektor-nedgång: {filtered_out.get('sector_downtrend', 0)}")
    
    print(f"\n🚨 Hard Stop Loss Triggers: {len(stop_loss_triggers)}")
    
    if len(stop_loss_triggers) > 0:
        print("\nHard Stop Loss Details:")
        for trigger in stop_loss_triggers:
            print(f"  {trigger['date'].strftime('%Y-%m-%d')} | {trigger['stock']:12} | {trigger['loss']:6.1f}% | Held {trigger['days_held']} days")
    
    return portfolio_series, buy_markers, sell_markers, partial_exit_markers, trades_df, cash

print("\n" + "="*70)
print("🚀 Kör backtest...")
print("="*70)

portfolio_series, buy_markers, sell_markers, partial_exit_markers, trades_df, cash = robust_backtest(
    signals, prices, prices_raw, volatility, volumes
)

# -----------------------------
# UTÖKAD ANALYS
# -----------------------------
def detailed_metrics(portfolio_series, trades_df, start_capital):
    """Utökad analys med fokus på win/loss ratio"""
    
    # Grundläggande metrics
    total_return = (portfolio_series.iloc[-1] / start_capital - 1) * 100
    years = len(portfolio_series) / 252
    annual_return = ((portfolio_series.iloc[-1] / start_capital) ** (1/years) - 1) * 100
    
    returns = portfolio_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    
    # Trade analys
    sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
    
    if len(sell_trades) > 0:
        winners = sell_trades[sell_trades['profit_pct'] > 0]
        losers = sell_trades[sell_trades['profit_pct'] < 0]
        
        win_rate = len(winners) / len(sell_trades) * 100
        avg_win = winners['profit_pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['profit_pct'].mean() if len(losers) > 0 else 0
        
        # Profit factor: total vinster / total förluster
        total_wins = winners['profit_pct'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['profit_pct'].sum()) if len(losers) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Exit reason analys
        exit_reasons = sell_trades['reason'].value_counts()
        
        # Stop loss trades
        stop_loss_trades = sell_trades[sell_trades['reason'].str.contains('Stop Loss')]
        trailing_stop_trades = sell_trades[sell_trades['reason'].str.contains('Trailing Stop')]
        
        metrics = {
            'Total Return': f"{total_return:.2f}%",
            'Annual Return': f"{annual_return:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Final Capital': f"{portfolio_series.iloc[-1]:,.0f} SEK",
            '---': '---',
            'Total Trades': len(sell_trades),
            'Win Rate': f"{win_rate:.1f}%",
            'Winners': len(winners),
            'Losers': len(losers),
            '----': '----',
            'Avg Win': f"{avg_win:.2f}%",
            'Avg Loss': f"{avg_loss:.2f}%",
            'Profit Factor': f"{profit_factor:.2f}",
            '-----': '-----',
            'Stop Losses': len(stop_loss_trades),
            'Trailing Stops': len(trailing_stop_trades)
        }
        
        return metrics, exit_reasons
    else:
        return {
            'Total Return': f"{total_return:.2f}%",
            'Note': 'Inga trades genomförda'
        }, None

metrics, exit_reasons = detailed_metrics(portfolio_series, trades_df, START_CAPITAL)

# Get current open positions (at end of backtest)
current_positions = {}
if len(trades_df) > 0:
    buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
    sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
    
    for _, buy in buy_trades.iterrows():
        stock = buy['stock']
        # Check if this position was sold
        stock_sells = sell_trades[sell_trades['stock'] == stock]
        stock_sells = stock_sells[stock_sells['date'] > buy['date']]
        
        if len(stock_sells) == 0 or stock_sells.iloc[-1]['portion'] == 0.5:
            # Position still open (or partially open)
            last_price = prices.loc[portfolio_series.index[-1], stock]
            last_z = signals.loc[portfolio_series.index[-1], stock]
            days_held = (portfolio_series.index[-1] - buy['date']).days
            
            entry_price = buy['price']
            current_pnl = (last_price / entry_price - 1) * 100
            
            # Check if partially closed
            partial = len(stock_sells) > 0 and stock_sells.iloc[-1]['portion'] == 0.5
            
            # Calculate position value
            shares_held = buy['shares']
            if partial:
                shares_held = shares_held * 0.5  # Only 50% left after partial exit
            
            position_value = shares_held * last_price
            
            current_positions[stock] = {
                'entry_date': buy['date'],
                'entry_price': entry_price,
                'entry_z': buy['z_score'],
                'current_price': last_price,
                'current_z': last_z,
                'pnl_pct': current_pnl,
                'days_held': days_held,
                'partial': partial,
                'shares': shares_held,
                'position_value': position_value
            }

print("\n" + "="*50)
print("📈 PRESTATIONSMÅTT")
print("="*50)
for key, value in metrics.items():
    if '---' in key:
        print()
    else:
        print(f"{key:.<30} {value}")

# Show current open positions
if len(current_positions) > 0:
    print("\n" + "="*50)
    print("📍 ÖPPNA POSITIONER (vid backtest slut)")
    print("="*50)
    
    total_invested = sum(pos['position_value'] for pos in current_positions.values())
    
    for stock, pos in current_positions.items():
        status = "50% SÅLD" if pos['partial'] else "FULL"
        color = "🟢" if pos['pnl_pct'] > 0 else "🔴"
        print(f"{color} {stock:12} | {status:10} | Värde: {pos['position_value']:>10,.0f} SEK | "
              f"P&L: {pos['pnl_pct']:6.2f}% | Dagar: {pos['days_held']:3} | "
              f"z: {pos['current_z']:5.2f} | Entry: {pos['entry_price']:.2f} → Nu: {pos['current_price']:.2f}")
    
    print("-" * 50)
    print(f"💰 TOTALT INVESTERAT: {total_invested:,.0f} SEK")
    print(f"💵 KVAR I CASH: {cash:,.0f} SEK")
    print(f"📊 PORTFÖLJVÄRDE: {portfolio_series.iloc[-1]:,.0f} SEK")

if exit_reasons is not None:
    print("\n" + "="*50)
    print("🚪 EXIT REASON FÖRDELNING")
    print("="*50)
    for reason, count in exit_reasons.items():
        print(f"{reason:.<40} {count}")

# Visa största förluster för analys
if len(trades_df) > 0:
    sell_trades = trades_df[trades_df['action'] == 'SELL']
    worst_trades = sell_trades.nsmallest(5, 'profit_pct')[['date', 'stock', 'profit_pct', 'reason']]
    
    print("\n" + "="*50)
    print("💔 VÄRSTA 5 TRADES (för analys)")
    print("="*50)
    for _, trade in worst_trades.iterrows():
        print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['stock']:12} | {trade['profit_pct']:6.2f}% | {trade['reason']}")

# (Copy ALL your data loading and backtest code here - everything BEFORE "DASH APP")
# Keep all the parameters, stock lists, data downloads, backtest function, etc.

# At the END, save the computed data:
data_to_save = {
    'portfolio_series': portfolio_series,
    'signals': signals,
    'prices': prices,
    'prices_raw': prices_raw,
    'volatility': volatility,
    'volumes': volumes,
    'trades_df': trades_df,
    'buy_markers': buy_markers,
    'sell_markers': sell_markers,
    'partial_exit_markers': partial_exit_markers,
    'current_positions': current_positions,
    'metrics': metrics,
    'cash': cash,
    'stocks': stocks,
    'sectors': sectors
}

with open('backtest_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)


print("✓ Backtest data saved!")
