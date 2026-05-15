"""
Modell 1: Maskininlärd prisanalys
Gymnasiearbete - Olle Kilander, TE23D

Strategi:
- Hämtar historiska aktiepriser för OMXS30 via yFinance
- Beräknar tekniska indikatorer: SMA(10), SMA(20), RSI(14)
- Tränar en Random Forest Classifier på data 2006-2016
- Testar modellen på data 2016-2025 (live-simulering)
- Jämför portföljutveckling mot OMXS30-index (buy & hold)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────
# INSTÄLLNINGAR
# ─────────────────────────────────────────
TRAIN_START = "2006-01-01"
TRAIN_END   = "2016-01-01"
TEST_START  = "2016-01-01"
TEST_END    = "2025-12-01"  # Rapporten: "perioden 2016-december 2025"

SMA_SHORT   = 10
SMA_LONG    = 20
RSI_PERIOD  = 14

INITIAL_CAPITAL = 100_000  # SEK

# OMXS30-aktier (tickers på Yahoo Finance)
OMXS30_TICKERS = [
    "ABB.ST",  "ALFA.ST", "ALIV-SDB.ST", "ASSA-B.ST", "ATCO-A.ST",
    "ATCO-B.ST","AZN.ST",  "BOL.ST",  "CAST.ST",  "ELUX-B.ST",
    "ERIC-B.ST","EVO.ST",  "GETI-B.ST","HEXA-B.ST","HM-B.ST",
    "INVE-B.ST","KINV-B.ST","LOOMIS.ST","NDA-SE.ST","NIBE-B.ST",
    "NOKIA.ST", "SAND.ST", "SCA-B.ST", "SEB-A.ST", "SECU-B.ST",
    "SKA-B.ST", "SKF-B.ST","SSAB-A.ST","SWED-A.ST","VOLV-B.ST",
]

# Index för jämförelse (buy & hold)
INDEX_TICKER = "^OMX"


# ─────────────────────────────────────────
# 1. DATAHÄMTNING
# ─────────────────────────────────────────
def hamta_data(tickers, start, end):
    """Hämtar och sorterar prisdata för en lista aktier via yFinance."""
    print(f"Hämtar data {start} → {end} för {len(tickers)} aktier...")
    raw = yf.download(tickers, start=start, end=end, interval="1d",  # daglig upplösning (1d) per rapporten
                      auto_adjust=True, progress=False)

    # Om flera tickers → MultiIndex; välj 'Close'
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers

    close = close.sort_index().dropna(how="all")
    print(f"  → {len(close)} handelsdagar, {close.shape[1]} aktier")
    return close


# ─────────────────────────────────────────
# 2. TEKNISKA INDIKATORER
# ─────────────────────────────────────────
def berakna_sma(close: pd.Series, period: int) -> pd.Series:
    """
    Glidande medelvärde (Simple Moving Average):
        SMA_n(t) = (1/n) * Σ P(t-i)  för i = 0..n-1
    """
    return close.rolling(window=period).mean()


def berakna_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (J. Welles Wilder):
        delta   = P(t) - P(t-1)
        RS      = avg_gain / avg_loss  (Wilder smoothing)
        RSI     = 100 - 100 / (1 + RS)
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder's smoothing (EWM med alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bygg_features(close: pd.Series) -> pd.DataFrame:
    """
    Bygger feature-DataFrame för en enskild aktie:
      - SMA_short, SMA_long
      - SMA_crossover: 1 om kort > lång (köpsignal), 0 annars
      - RSI
      - Target: 1 om nästa dags stängning > dagens (upp), 0 annars
    """
    sma_s = berakna_sma(close, SMA_SHORT)
    sma_l = berakna_sma(close, SMA_LONG)
    rsi   = berakna_rsi(close, RSI_PERIOD)

    # Korsningssignal: subtrahera paren med varandra och jämför mot gårdagen
    diff_today     = sma_s - sma_l
    diff_yesterday = diff_today.shift(1)
    sma_cross = ((diff_yesterday < 0) & (diff_today > 0)).astype(int)

    df = pd.DataFrame({
        "close":    close,
        "sma_s":    sma_s,
        "sma_l":    sma_l,
        "sma_cross": sma_cross,
        "rsi":       rsi,
    })

    # Målvariabel: gick aktien upp nästa dag?
    df["target"] = (close.shift(-1) > close).astype(int)

    return df.dropna()


# ─────────────────────────────────────────
# 3. TRÄNING & PREDIKTION (RANDOM FOREST)
# ─────────────────────────────────────────
FEATURES = ["sma_s", "sma_l", "sma_cross", "rsi"]

def trana_modell(df_train: pd.DataFrame) -> RandomForestClassifier:
    """Tränar en Random Forest på träningsdata."""
    X = df_train[FEATURES]
    y = df_train["target"]

    model = RandomForestClassifier(
        n_estimators=200,       # antal beslutsträd
        max_depth=5,            # begränsar djup för att minska överanpassning
        min_samples_leaf=20,    # kräver minst 20 datapunkter per löv
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predicera(model: RandomForestClassifier, df_test: pd.DataFrame) -> pd.Series:
    """Returnerar prediktion (0 eller 1) för testdata."""
    X = df_test[FEATURES]
    return pd.Series(model.predict(X), index=df_test.index, name="signal")


# ─────────────────────────────────────────
# 4. BACKTESTING / HANDELSSIMULERING
# ─────────────────────────────────────────
def simulera_handel(df_test: pd.DataFrame, signal: pd.Series,
                    kapital: float) -> pd.DataFrame:
    """
    Long-only simulering med tre möjliga tillstånd per rapporten:
      - Köp  (signal=1): investera allt kapital i aktien
      - Sälj (signal=0): sälj om position finns, håll kontanter
      - Ingen position : om signal=0 och ingen position → gör ingenting
    Portföljvärde uppdateras löpande vid varje tidssteg (dag).
    Transaktioner sker till aktuellt marknadspris (stängningspris).
    """
    close   = df_test["close"]
    resultat = []
    position = 0     # antal aktier innehavda
    cash     = kapital

    for i in range(len(df_test) - 1):
        dag   = df_test.index[i]
        pris  = close.iloc[i]
        sig   = signal.iloc[i]

        if sig == 1 and cash > 0:
            # Köp
            position = cash / pris
            cash     = 0.0
        elif sig == 0 and position > 0:
            # Sälj
            cash     = position * pris
            position = 0.0

        portfolio_val = cash + position * pris
        resultat.append({"datum": dag, "portfolio": portfolio_val})

    return pd.DataFrame(resultat).set_index("datum")


# ─────────────────────────────────────────
# 5. JÄMFÖRELSEINDEX (BUY & HOLD)
# ─────────────────────────────────────────
def buy_and_hold(close: pd.Series, kapital: float) -> pd.Series:
    """Beräknar buy-and-hold portföljvärde för ett givet index/aktie."""
    andelar = kapital / close.iloc[0]
    return (close * andelar).rename("index_bah")


# ─────────────────────────────────────────
# 6. STATISTIK
# ─────────────────────────────────────────
def berakna_statistik(portfölj: pd.Series, namn: str = "Bot") -> dict:
    daglig_avk = portfölj.pct_change().dropna()
    total_avk  = (portfölj.iloc[-1] / portfölj.iloc[0] - 1) * 100
    ar_avk     = ((portfölj.iloc[-1] / portfölj.iloc[0]) **
                  (252 / len(portfölj)) - 1) * 100

    # Sharpe (riskfri ränta ≈ 0)
    sharpe = (daglig_avk.mean() / daglig_avk.std()) * np.sqrt(252) if daglig_avk.std() > 0 else 0

    # Max drawdown
    rullande_max = portfölj.cummax()
    drawdown     = (portfölj - rullande_max) / rullande_max
    max_dd       = drawdown.min() * 100

    stats = {
        "Namn":              namn,
        "Totalavkastning %": round(total_avk, 2),
        "Årsavkastning %":   round(ar_avk, 2),
        "Sharpe":            round(sharpe, 3),
        "Max Drawdown %":    round(max_dd, 2),
    }
    print(f"\n── {namn} ──")
    for k, v in stats.items():
        if k != "Namn":
            print(f"  {k}: {v}")
    return stats


# ─────────────────────────────────────────
# 7. VISUALISERING
# ─────────────────────────────────────────
def visa_resultat(bot_portföljer: dict, index_bah: pd.Series):
    """Ritar portföljutveckling med Plotly."""
    fig = go.Figure()

    for namn, serie in bot_portföljer.items():
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.values,
            mode="lines", name=namn
        ))

    fig.add_trace(go.Scatter(
        x=index_bah.index, y=index_bah.values,
        mode="lines", name="OMXS30 Buy & Hold",
        line=dict(dash="dash", color="black", width=2)
    ))

    fig.update_layout(
        title="Modell 1: AI-bot vs OMXS30 Buy & Hold (2016–2025)",
        xaxis_title="Datum",
        yaxis_title="Portföljvärde (SEK)",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )
    fig.show()


# ─────────────────────────────────────────
# 8. HUVUDPROGRAM
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Modell 1 – Maskininlärd prisanalys (Random Forest)")
    print("=" * 55)

    # --- Hämta data ---
    train_close = hamta_data(OMXS30_TICKERS, TRAIN_START, TRAIN_END)
    test_close  = hamta_data(OMXS30_TICKERS, TEST_START,  TEST_END)

    # Hämta index för buy & hold-jämförelse
    index_raw  = yf.download(INDEX_TICKER, start=TEST_START,
                              end=TEST_END, auto_adjust=True, progress=False)
    index_pris = index_raw["Close"].squeeze()
    index_bah  = buy_and_hold(index_pris, INITIAL_CAPITAL)

    # --- Loopa över aktier, träna & testa ---
    bot_portföljer = {}

    for ticker in OMXS30_TICKERS:
        if ticker not in train_close.columns or ticker not in test_close.columns:
            continue

        # Bygg features
        df_train = bygg_features(train_close[ticker])
        df_test  = bygg_features(test_close[ticker])

        if len(df_train) < 200 or len(df_test) < 50:
            continue  # för lite data

        # Träna modell
        model = trana_modell(df_train)

        # Prediktera på testdata
        signal = predicera(model, df_test)

        acc = accuracy_score(df_test["target"], signal)
        print(f"{ticker}: träffsäkerhet = {acc:.1%}")

        # Simulera handel
        portfolio = simulera_handel(df_test, signal, INITIAL_CAPITAL)
        bot_portföljer[ticker] = portfolio["portfolio"]

    if not bot_portföljer:
        print("Ingen aktie kunde simuleras. Kontrollera datahämtningen.")
        return

    # Kombinera till gemensam portfölj med lika kapitalvikt från start
    # (inte snitt av separata portföljer, utan normaliserad gemensam utveckling)
    portfölj_df  = pd.DataFrame(bot_portföljer).dropna()
    # Normalisera varje aktie till 1.0 vid start, ta medel = gemensam avkastning
    normaliserad    = portfölj_df.div(portfölj_df.iloc[0])
    snitt_portfolio = (normaliserad.mean(axis=1) * INITIAL_CAPITAL)

    # --- Statistik ---
    berakna_statistik(snitt_portfolio, "AI-bot (snitt OMXS30)")
    berakna_statistik(index_bah,       "OMXS30 Buy & Hold")

    # --- Visualisering ---
    visa_resultat({"AI-bot (snitt)": snitt_portfolio}, index_bah)

    print("\nKlart!")


if __name__ == "__main__":
    main()
