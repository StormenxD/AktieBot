import pickle
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
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
# Color scheme
dark_bg = "#0e0e0e"
text_color = "#e0e0e0"
accent = "#00ff7f"
danger = "#ff4d4d"
warning = "#ffa500"

# Load precomputed data
with open('backtest_data.pkl', 'rb') as f:
    data = pickle.load(f)

portfolio_series = data['portfolio_series']
signals = data['signals']
prices = data['prices']
trades_df = data['trades_df']
buy_markers = data['buy_markers']
sell_markers = data['sell_markers']
partial_exit_markers = data['partial_exit_markers']
current_positions = data['current_positions']
metrics = data['metrics']
cash = data['cash']
stocks = data['stocks']
sectors = data['sectors']

# Your entire DASH APP code here (no changes needed)
app = Dash(__name__)
app.title = "Pair Trading"

stock_options = [{"label": s, "value": s} for s in stocks.keys()]
sector_options = [{"label": s, "value": s} for s in sectors.keys()]

app.layout = html.Div(
    style={"backgroundColor": dark_bg, "color": text_color, "padding": "20px", "fontFamily": "Arial"},
    children=[
        html.H2("🎯 Pair Trading – OMXS30 + NASDAQ Top 50", style={"textAlign": "center"}),
        
        # Prestationsmått
        html.Div([
            html.H3("📊 Prestationsmått", style={"color": accent}),
            html.Div([
                html.Div([
                    html.Strong(f"{key}: " if not '---' in key else ""),
                    html.Span(value if not '---' in key else "")
                ], style={"marginBottom": "8px" if not '---' in key else "0px"})
                for key, value in metrics.items()
            ])
        ], style={
            "backgroundColor": "#1a1a1a",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "20px"
        }),
        
        # NYTT: Öppna Positioner
        html.Div([
            html.H3("📍 Öppna Positioner (vid backtest slut)", style={"color": warning}),
            html.Div([
                html.Div([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Aktie", style={"padding": "8px", "textAlign": "left"}),
                                html.Th("Status", style={"padding": "8px"}),
                                html.Th("Positionsvärde", style={"padding": "8px"}),
                                html.Th("Entry Datum", style={"padding": "8px"}),
                                html.Th("Entry Pris", style={"padding": "8px"}),
                                html.Th("Nuvarande Pris", style={"padding": "8px"}),
                                html.Th("P&L %", style={"padding": "8px"}),
                                html.Th("Dagar Hållen", style={"padding": "8px"}),
                                html.Th("Entry Z", style={"padding": "8px"}),
                                html.Th("Nuvarande Z", style={"padding": "8px"}),
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(stock, style={"padding": "8px", "fontWeight": "bold"}),
                                html.Td(
                                    "🔶 50% SÅLD" if pos['partial'] else "🟢 FULL",
                                    style={"padding": "8px", "textAlign": "center"}
                                ),
                                html.Td(
                                    f"{pos['position_value']:,.0f} SEK",
                                    style={"padding": "8px", "fontWeight": "bold", "color": "#FFD700"}
                                ),
                                html.Td(
                                    pos['entry_date'].strftime('%Y-%m-%d'),
                                    style={"padding": "8px"}
                                ),
                                html.Td(
                                    f"{pos['entry_price']:.2f} SEK",
                                    style={"padding": "8px"}
                                ),
                                html.Td(
                                    f"{pos['current_price']:.2f} SEK",
                                    style={"padding": "8px"}
                                ),
                                html.Td(
                                    f"{pos['pnl_pct']:.2f}%",
                                    style={
                                        "padding": "8px",
                                        "color": accent if pos['pnl_pct'] > 0 else danger,
                                        "fontWeight": "bold"
                                    }
                                ),
                                html.Td(
                                    f"{pos['days_held']} dagar",
                                    style={"padding": "8px"}
                                ),
                                html.Td(
                                    f"{pos['entry_z']:.2f}",
                                    style={"padding": "8px"}
                                ),
                                html.Td(
                                    f"{pos['current_z']:.2f}",
                                    style={
                                        "padding": "8px",
                                        "color": accent if pos['current_z'] > pos['entry_z'] else warning
                                    }
                                ),
                            ]) for stock, pos in current_positions.items()
                        ]) if len(current_positions) > 0 else html.Tbody([
                            html.Tr([
                                html.Td(
                                    "🎉 Inga öppna positioner - allt har stängts!",
                                    colSpan=10,
                                    style={"padding": "20px", "textAlign": "center", "fontStyle": "italic"}
                                )
                            ])
                        ])
                    ], style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                        "border": f"1px solid {text_color}"
                    }),
                    # Summary statistics
                    html.Div([
                        html.Div([
                            html.Strong("💰 Totalt Investerat: "),
                            html.Span(
                                f"{sum(pos['position_value'] for pos in current_positions.values()):,.0f} SEK" 
                                if len(current_positions) > 0 else "0 SEK",
                                style={"color": "#FFD700", "fontSize": "18px", "fontWeight": "bold"}
                            )
                        ], style={"marginTop": "15px", "marginBottom": "8px"}),
                        html.Div([
                            html.Strong("💵 Cash Kvar: "),
                            html.Span(
                                f"{cash:,.0f} SEK",
                                style={"color": accent, "fontSize": "16px"}
                            )
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Strong("📊 Totalt Portföljvärde: "),
                            html.Span(
                                f"{portfolio_series.iloc[-1]:,.0f} SEK",
                                style={"color": accent, "fontSize": "18px", "fontWeight": "bold"}
                            )
                        ], style={"marginBottom": "8px"}),
                    ], style={"marginTop": "20px", "padding": "15px", "backgroundColor": "#0a0a0a", "borderRadius": "5px"})
                ])
            ], style={"overflowX": "auto"})
        ], style={
            "backgroundColor": "#1a1a1a",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "20px",
            "border": f"2px solid {warning}"
        }),
        
        # Skyddsåtgärder
        html.Div([
            html.H3("🛡️ Skyddsåtgärder mot Förluster", style={"color": danger}),
            html.Div([
                html.Div([
                    html.Strong("🚨 Hard Stop Loss: "),
                    html.Span(f"-{STOP_LOSS_PCT*100:.0f}% (HÖGSTA PRIORITET)")
                ], style={"marginBottom": "8px", "color": danger, "fontSize": "16px"}),
                html.Div([
                    html.Strong("Z-Score Stop Loss: "),
                    html.Span(f"z < {STOP_LOSS_Z}")
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("Trailing Stop: "),
                    html.Span(f"{TRAILING_STOP_PCT*100:.0f}% från högsta värde")
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("Partial Exits: "),
                    html.Span(f"Ta hem 50% vinst vid z > {PARTIAL_EXIT_Z}")
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("Volatilitetsfilter: "),
                    html.Span(f"Undvik aktier med >{MAX_VOLATILITY*100:.0f}% daglig volatilitet")
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Strong("🆕 Sektor-Trend Filter: "),
                    html.Span(f"Skip köp om sektor ned >{abs(SECTOR_TREND_THRESHOLD)*100:.0f}% över {SECTOR_TREND_DAYS} dagar")
                ], style={"marginBottom": "8px", "color": accent}),
                html.Div([
                    html.Strong("Emergency Exits: "),
                    html.Span(f"Exit vid {MAX_DAYS_NO_DATA} dagar utan data ELLER {MAX_DAYS_IN_TRADE} dagar i trade")
                ], style={"marginBottom": "8px"}),
            ])
        ], style={
            "backgroundColor": "#1a1a1a",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "20px",
            "border": f"2px solid {danger}"
        }),
        
        # Kontroller
        html.Div([
            html.Div([
                html.Label("Välj sektor:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="sector-dropdown",
                    options=sector_options,
                    value=None,
                    style={"color": "#000"}
                )
            ], style={"width": "40%", "display": "inline-block", "marginRight": "20px"}),

            html.Div([
                html.Label("Välj aktier:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="stock-dropdown",
                    options=stock_options,
                    value=[stock_options[0]["value"]],
                    multi=True,
                    style={"color": "#000"}
                )
            ], style={"width": "55%", "display": "inline-block"}),
        ], style={"marginBottom": "30px"}),

        dcc.Graph(id="equity-chart"),
        dcc.Graph(id="zscore-chart"),
        dcc.Graph(id="trade-analysis-chart")
    ]
)

# -----------------------------
# CALLBACKS
# -----------------------------
@app.callback(
    Output("stock-dropdown", "options"),
    Output("stock-dropdown", "value"),
    Input("sector-dropdown", "value")
)
def filter_stocks_by_sector(sector):
    if sector is None:
        options = [{"label": s, "value": s} for s in stocks.keys()]
        return options, [options[0]["value"]]
    options = [{"label": s, "value": s} for s in sectors[sector]]
    return options, [options[0]["value"]]

@app.callback(
    Output("equity-chart", "figure"),
    Output("zscore-chart", "figure"),
    Output("trade-analysis-chart", "figure"),
    Input("stock-dropdown", "value")
)
def update_graphs(selected_stocks):
    if not selected_stocks:
        return go.Figure(), go.Figure(), go.Figure()

    # Equity curve
    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(
        x=portfolio_series.index,
        y=portfolio_series.values,
        mode="lines",
        name="Pair Trading",
        line=dict(width=3, color=accent)
    ))

    # OMXS30 Index
    if "^OMX" in prices.columns:
        index_returns = (prices["^OMX"].pct_change()[WINDOW:] + 1).cumprod() * START_CAPITAL
        fig_equity.add_trace(go.Scatter(
            x=index_returns.index,
            y=index_returns.values,
            mode="lines",
            name="OMXS30 Index",
            line=dict(width=2, color=danger, dash='dash')
        ))
    
    # NASDAQ Composite Index
    if "^IXIC" in prices.columns:
        nasdaq_returns = (prices["^IXIC"].pct_change()[WINDOW:] + 1).cumprod() * START_CAPITAL
        fig_equity.add_trace(go.Scatter(
            x=nasdaq_returns.index,
            y=nasdaq_returns.values,
            mode="lines",
            name="NASDAQ Composite",
            line=dict(width=2, color="#00D9FF", dash='dot')
        ))

    for stock in selected_stocks:
        stock_values = (prices[stock].pct_change()[WINDOW:] + 1).cumprod() * START_CAPITAL
        fig_equity.add_trace(go.Scatter(
            x=stock_values.index,
            y=stock_values.values,
            mode="lines",
            name=f"{stock} Buy & Hold",
            line=dict(dash="dot", width=1),
            opacity=0.5
        ))

    fig_equity.update_layout(
        title="Portföljvärde med Robust Risk Management",
        xaxis_title="Datum",
        yaxis_title="Kapital (SEK)",
        template="plotly_dark",
        hovermode="x unified"
    )

    # Z-score med alla markörer
    fig_z = go.Figure()
    
    for stock in selected_stocks:
        fig_z.add_trace(go.Scatter(
            x=signals.index,
            y=signals[stock],
            mode="lines",
            name=f"{stock} Z-score",
            line=dict(width=2)
        ))

        # Köp markörer
        if stock in buy_markers and len(buy_markers[stock]) > 0:
            dates, z_scores = zip(*buy_markers[stock])
            fig_z.add_trace(go.Scatter(
                x=dates, y=z_scores,
                mode="markers",
                marker=dict(color="lime", size=12, symbol="triangle-up", line=dict(width=2, color="darkgreen")),
                name=f"{stock} Köp"
            ))

        # Partial exit markörer
        if stock in partial_exit_markers and len(partial_exit_markers[stock]) > 0:
            dates, z_scores = zip(*partial_exit_markers[stock])
            fig_z.add_trace(go.Scatter(
                x=dates, y=z_scores,
                mode="markers",
                marker=dict(color="orange", size=10, symbol="diamond", line=dict(width=2, color="darkorange")),
                name=f"{stock} Partial Exit (50%)"
            ))

        # Sälj markörer
        if stock in sell_markers and len(sell_markers[stock]) > 0:
            dates, z_scores = zip(*sell_markers[stock])
            fig_z.add_trace(go.Scatter(
                x=dates, y=z_scores,
                mode="markers",
                marker=dict(color="red", size=12, symbol="triangle-down", line=dict(width=2, color="darkred")),
                name=f"{stock} Sälj (Full Exit)"
            ))

    # Referenslinjer
    fig_z.add_hline(y=ENTRY_Z, line_dash="dash", line_color="lime", 
                    annotation_text=f"Entry ({ENTRY_Z})")
    fig_z.add_hline(y=PARTIAL_EXIT_Z, line_dash="dash", line_color="orange",
                    annotation_text=f"Partial Exit ({PARTIAL_EXIT_Z})")
    fig_z.add_hline(y=FULL_EXIT_Z, line_dash="dash", line_color="yellow",
                    annotation_text=f"Full Exit ({FULL_EXIT_Z})")
    fig_z.add_hline(y=STOP_LOSS_Z, line_dash="dot", line_color="red",
                    annotation_text=f"Stop Loss ({STOP_LOSS_Z})")
    fig_z.add_hline(y=0, line_color="gray", line_width=1)
    
    fig_z.update_layout(
        title="Z-score med Multi-Level Exit Strategi",
        xaxis_title="Datum",
        yaxis_title="Z-score",
        template="plotly_dark",
        hovermode="x unified"
    )

    # Trade Analysis
    fig_trades = go.Figure()
    
    if len(trades_df) > 0:
        sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        if len(sell_trades) > 0:
            # Scatter plot av profit per trade
            colors = ['green' if x > 0 else 'red' for x in sell_trades['profit_pct']]
            
            fig_trades.add_trace(go.Scatter(
                x=sell_trades['date'],
                y=sell_trades['profit_pct'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(width=1, color='white')
                ),
                text=sell_trades['stock'] + '<br>' + sell_trades['reason'],
                name='Trades'
            ))
            
            fig_trades.add_hline(y=0, line_color="gray", line_width=2)
            fig_trades.add_hline(y=sell_trades['profit_pct'].mean(), 
                               line_dash="dash", line_color="white",
                               annotation_text=f"Genomsnitt: {sell_trades['profit_pct'].mean():.2f}%")
    
    fig_trades.update_layout(
        title="Trade Performance över Tid",
        xaxis_title="Datum",
        yaxis_title="Profit (%)",
        template="plotly_dark",
        hovermode="closest"
    )

    return fig_equity, fig_z, fig_trades


if __name__ == "__main__":
    print("\n🚀 Startar Dash app på http://127.0.0.1:8050")
    print("Tryck Ctrl+C för att avsluta\n")
    app.run(debug=False)