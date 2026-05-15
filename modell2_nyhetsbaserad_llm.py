"""
Modell 2: Nyhetsbaserad riskanalys med Large Language Models (LLM)
Gymnasiearbete - Olle Kilander, TE23D

OBS: Denna kod är delvis rekonstruerad efter dataförlust (SSD-haveri).
     Koden är därför inte i slutligt skick.

Strategi:
- Skrapar nyhetsartiklar från Dagens Industri, Placera.se,
  Wall Street Journal och Donald Trumps X-konto
- Skickar artikeltext till Llama via API för sentimentanalys
- Llama klassificerar: positiv/negativ + uppskattar framtida pris (Pt+1 = Pt + ΔP)
- Handlar baserat på prediktionen och jämför mot OMXS30 buy & hold

Begränsningar (från rapporten):
- Llama 4 API blev betalversion under projektet → modellen aldrig färdigställd
- HTML-skrapning ger stökig text som kan påverka Llamas tolkningar
- Kan bara använda sidor där artikeln är lätt att hitta i HTML-koden
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import time
import json
import re
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# INSTÄLLNINGAR
# ─────────────────────────────────────────
TEST_START = "2024-01-01"
TEST_END   = "2025-12-01"

INITIAL_CAPITAL = 100_000  # SEK

# ── LLM-val: välj ett av alternativen nedan ──────────────────────────
#
# ALTERNATIV 1: Groq (gratis molntjänst, kräver API-nyckel)
#   1. Skapa gratis konto på https://console.groq.com
#   2. Generera en API-nyckel under "API Keys"
#   3. Klistra in nyckeln nedan
#
LLM_BACKEND   = "groq"                                          # "groq" eller "ollama"
LLAMA_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLAMA_API_KEY = "DIN_GROQ_NYCKEL_HÄR"                          # ← ersätt
LLAMA_MODEL   = "llama3-70b-8192"                               # gratis på Groq
#
# ALTERNATIV 2: Ollama (helt lokalt, gratis, ingen nyckel)
#   1. Installera Ollama: https://ollama.com/download
#   2. Kör i terminalen: ollama pull llama3
#   3. Sätt LLM_BACKEND = "ollama" nedan
#
# LLM_BACKEND   = "ollama"
# LLAMA_API_URL = "http://localhost:11434/api/chat"
# LLAMA_API_KEY = ""          # krävs inte för Ollama
# LLAMA_MODEL   = "llama3"
# ─────────────────────────────────────────────────────────────────────

# OMXS30-aktier att följa
OMXS30_TICKERS = [
    "ABB.ST",  "ALFA.ST", "ASSA-B.ST", "AZN.ST",  "ERIC-B.ST",
    "EVO.ST",  "HEXA-B.ST","HM-B.ST",  "NDA-SE.ST","NIBE-B.ST",
    "SAND.ST", "SEB-A.ST", "SKF-B.ST", "SWED-A.ST","VOLV-B.ST",
]

INDEX_TICKER = "^OMX"

# Tidsgräns för hur gammal en artikel får vara (dagar)
MAX_ARTIKEL_ÅLDER_DAGAR = 1


# ─────────────────────────────────────────
# 1. NYHETSINHÄMTNING (DATASCRAPING)
# ─────────────────────────────────────────
# HTML-skrapning: importera HTML-kod och leta upp artikeltexten.
# Kan bara användas på sidor där artikeln är lätt att hitta i HTML-koden.
# De flesta stora nyhetssidor kan därför inte användas.

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def skrapa_dagens_industri(max_artiklar: int = 5) -> list[dict]:
    """
    Skrapar senaste nyheter från Dagens Industri.
    Returnerar lista med {titel, text, kalla}.
    """
    artiklar = []
    try:
        url  = "https://www.di.se/nyheter/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Hitta artikellänkar
        lankar = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/nyheter/" in href and len(href) > 20:
                full = href if href.startswith("http") else "https://www.di.se" + href
                if full not in lankar:
                    lankar.append(full)

        for lank in lankar[:max_artiklar]:
            try:
                art_resp = requests.get(lank, headers=HEADERS, timeout=10)
                art_soup = BeautifulSoup(art_resp.text, "html.parser")

                # Artikeltext finns oftast i <article> eller <p>-taggar
                artikel_tag = art_soup.find("article")
                if not artikel_tag:
                    artikel_tag = art_soup

                stycken = artikel_tag.find_all("p")
                text    = " ".join(p.get_text(strip=True) for p in stycken)
                titel   = art_soup.find("h1")
                titel   = titel.get_text(strip=True) if titel else lank

                if len(text) > 100:
                    artiklar.append({"titel": titel, "text": text[:2000], "kalla": "Dagens Industri"})
            except Exception:
                continue

    except Exception as e:
        print(f"  DI-skrapning misslyckades: {e}")

    return artiklar


def skrapa_placera(max_artiklar: int = 5) -> list[dict]:
    """
    Skrapar senaste nyheter från Placera.se.
    """
    artiklar = []
    try:
        url  = "https://www.placera.se/placera/redaktionellt.html"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        lankar = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/placera/" in href and ".html" in href:
                full = href if href.startswith("http") else "https://www.placera.se" + href
                if full not in lankar and full != url:
                    lankar.append(full)

        for lank in lankar[:max_artiklar]:
            try:
                art_resp = requests.get(lank, headers=HEADERS, timeout=10)
                art_soup = BeautifulSoup(art_resp.text, "html.parser")

                stycken = art_soup.find_all("p")
                text    = " ".join(p.get_text(strip=True) for p in stycken)
                titel   = art_soup.find("h1")
                titel   = titel.get_text(strip=True) if titel else lank

                if len(text) > 100:
                    artiklar.append({"titel": titel, "text": text[:2000], "kalla": "Placera.se"})
            except Exception:
                continue

    except Exception as e:
        print(f"  Placera-skrapning misslyckades: {e}")

    return artiklar


def skrapa_wsj(max_artiklar: int = 3) -> list[dict]:
    """
    Skrapar rubriker från Wall Street Journal (marknadssektionen).
    OBS: WSJ kräver prenumeration för full text – endast rubrik + ingress hämtas.
    """
    artiklar = []
    try:
        url  = "https://www.wsj.com/news/markets"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        for h3 in soup.find_all(["h3", "h2"], limit=max_artiklar * 2):
            text = h3.get_text(strip=True)
            if len(text) > 20:
                artiklar.append({"titel": text, "text": text, "kalla": "Wall Street Journal"})
            if len(artiklar) >= max_artiklar:
                break

    except Exception as e:
        print(f"  WSJ-skrapning misslyckades: {e}")

    return artiklar


def skrapa_trump_x(max_inlagg: int = 5) -> list[dict]:
    """
    Hämtar Trumps senaste inlägg från X (Twitter).
    OBS: X blockerar de flesta skrapningsförsök utan autentisering.
    Använder Nitter (öppen frontend) som fallback om tillgänglig.
    """
    artiklar = []
    nitter_instanser = [
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
    ]

    for instans in nitter_instanser:
        try:
            url  = f"{instans}/realDonaldTrump"
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            tweets = soup.find_all("div", class_="tweet-content")

            for tweet in tweets[:max_inlagg]:
                text = tweet.get_text(strip=True)
                if len(text) > 10:
                    artiklar.append({"titel": "Trump X-inlägg", "text": text, "kalla": "Donald Trump X"})

            if artiklar:
                break

        except Exception:
            continue

    if not artiklar:
        print("  Trump X-skrapning misslyckades (X blockerar utan autentisering)")

    return artiklar


def hamta_alla_nyheter() -> list[dict]:
    """Hämtar nyheter från alla källor och returnerar kombinerad lista."""
    print("Hämtar nyheter...")
    alla = []
    alla += skrapa_dagens_industri()
    alla += skrapa_placera()
    alla += skrapa_wsj()
    alla += skrapa_trump_x()
    print(f"  → {len(alla)} artiklar/inlägg hämtade")
    return alla


# ─────────────────────────────────────────
# 2. LLM-ANALYS (LLAMA)
# ─────────────────────────────────────────
def sanera_text(text: str) -> str:
    """Tar bort tecken som inte kan hanteras av ASCII-begränsade bibliotek."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def analysera_med_llama(artikel: dict, aktuellt_pris: float, ticker: str) -> dict | None:
    """
    Skickar en nyhetsartikel till LLM för sentimentanalys.

    Llama instrueras att:
      1. Klassificera artikeln som positiv eller negativ ur marknadsperspektiv
      2. Ge en numerisk uppskattning av framtida rörelse: Pt+1 = Pt + ΔP

    Stödjer två backends: Groq (gratis moln) och Ollama (lokalt, ingen nyckel).
    Returnerar dict med {sentiment, delta_p, förväntat_pris} eller None vid fel.
    """
    prompt = f"""Du ar en finansanalytiker. Analysera foljande nyhetsartikel ur ett marknadsperspektiv for aktien {ticker} (nuvarande pris: {aktuellt_pris:.2f} SEK).

Artikel fran {sanera_text(artikel['kalla'])}:
"{sanera_text(artikel['text'][:1500])}"

Svara ENDAST med ett JSON-objekt i foljande format (ingen annan text):
{{
  "sentiment": "positiv" eller "negativ",
  "delta_p": <numerisk forandring i SEK, t.ex. 2.5 eller -1.8>,
  "motivering": "<kort motivering pa max 20 ord>"
}}"""

    try:
        if LLM_BACKEND == "ollama":
            # Ollama: lokalt API, ingen nyckel krävs
            payload = {
                "model":    LLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
            }
            resp     = requests.post(LLAMA_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            innehall = resp.json()["message"]["content"]

        else:
            # Groq via httpx (undviker redirect-problem med requests)
            import httpx
            headers = {
                "Authorization": f"Bearer {LLAMA_API_KEY}",
                "Content-Type":  "application/json",
            }
            payload = {
                "model":       LLAMA_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens":  200,
            }
            import json as _json
            with httpx.Client(follow_redirects=False) as client:
                resp = client.post(
                    LLAMA_API_URL,
                    headers={**headers, "Content-Type": "application/json; charset=utf-8"},
                    content=_json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=30,
                )
            resp.raise_for_status()
            innehall = resp.json()["choices"][0]["message"]["content"]

        # Extrahera JSON från svaret
        json_match = re.search(r'\{.*\}', innehall, re.DOTALL)
        if not json_match:
            return None

        data      = json.loads(json_match.group())
        delta_p   = float(data.get("delta_p", 0))
        sentiment = data.get("sentiment", "okänd")

        return {
            "sentiment":      sentiment,
            "delta_p":        delta_p,
            "förväntat_pris": aktuellt_pris + delta_p,  # Pt+1 = Pt + ΔP
            "motivering":     data.get("motivering", ""),
        }

    except Exception as e:
        print(f"    LLM-fel: {e}")
        return None


# ─────────────────────────────────────────
# 3. PREDIKTION & HANDELSSIGNAL
# ─────────────────────────────────────────
def generera_signal(analyser: list[dict], aktuellt_pris: float) -> tuple[int, float]:
    """
    Aggregerar flera Llama-analyser till en handelssignal.

    Modellen arbetar inte med binär klassificering utan med exakta prisprediktioner:
        Pt+1 = Pt + ΔP
    där ΔP härleds ur Llamas tolkning av nyhetspåverkan.

    Returnerar (signal, förväntat_pris):
      signal = 1  → köp (positivt sentiment, pris förväntas stiga)
      signal = -1 → sälj (negativt sentiment, pris förväntas falla)
      signal = 0  → ingen position
    """
    if not analyser:
        return 0, aktuellt_pris

    # Viktat medelvärde av ΔP från alla analyser
    delta_p_lista = [a["delta_p"] for a in analyser if a is not None]
    if not delta_p_lista:
        return 0, aktuellt_pris

    snitt_delta = np.mean(delta_p_lista)
    förväntat   = aktuellt_pris + snitt_delta

    # Sätt signal baserat på riktning och magnitude (minst 0.5% rörelse)
    tröskel = aktuellt_pris * 0.005
    if snitt_delta > tröskel:
        return 1, förväntat
    elif snitt_delta < -tröskel:
        return -1, förväntat
    else:
        return 0, förväntat


# ─────────────────────────────────────────
# 4. BACKTESTING / HANDELSSIMULERING
# ─────────────────────────────────────────
def simulera_handel(signaler: pd.DataFrame, pris_data: pd.DataFrame,
                    kapital: float) -> pd.Series:
    """
    Simulerar handel baserat på LLM-signaler.
    Tre möjliga tillstånd per dag och aktie:
      - Köp  (signal=1):  investera andel av kapital
      - Sälj (signal=-1): stäng position
      - Ingen position (signal=0): gör ingenting

    Portföljvärde uppdateras löpande vid varje tidssteg (dag).
    Alla transaktioner sker till aktuellt marknadspris (stängningspris).
    """
    portfölj  = kapital
    positioner = {}  # ticker → antal aktier
    historik   = []

    for dag in pris_data.index:
        # Uppdatera portföljvärde med aktuella priser
        dagsvärde = portfölj
        for ticker, antal in list(positioner.items()):
            if ticker in pris_data.columns and dag in pris_data.index:
                dagsvärde += antal * pris_data.loc[dag, ticker]

        historik.append({"datum": dag, "portfolio": dagsvärde})

        # Handla baserat på signaler
        if dag not in signaler.index:
            continue

        for ticker in pris_data.columns:
            if ticker not in signaler.columns:
                continue
            if pd.isna(pris_data.loc[dag, ticker]):
                continue

            pris   = pris_data.loc[dag, ticker]
            signal = signaler.loc[dag, ticker] if ticker in signaler.columns else 0

            if signal == 1 and ticker not in positioner and portfölj > 0:
                # Köp: max 20% av portfölj per aktie
                invest  = min(portfölj * 0.20, portfölj)
                antal   = invest / pris
                positioner[ticker] = antal
                portfölj -= invest

            elif signal == -1 and ticker in positioner:
                # Sälj: stäng position
                portfölj += positioner[ticker] * pris
                del positioner[ticker]

    df = pd.DataFrame(historik).set_index("datum")
    return df["portfolio"]


# ─────────────────────────────────────────
# 5. JÄMFÖRELSEINDEX (BUY & HOLD)
# ─────────────────────────────────────────
def buy_and_hold(close: pd.Series, kapital: float) -> pd.Series:
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

    sharpe = (daglig_avk.mean() / daglig_avk.std()) * np.sqrt(252) if daglig_avk.std() > 0 else 0

    rullande_max = portfölj.cummax()
    drawdown     = (portfölj - rullande_max) / rullande_max
    max_dd       = drawdown.min() * 100

    # Win rate: andel dagar med positiv avkastning
    win_rate = (daglig_avk > 0).sum() / len(daglig_avk) * 100

    stats = {
        "Namn":              namn,
        "Totalavkastning %": round(total_avk, 2),
        "Årsavkastning %":   round(ar_avk, 2),
        "Sharpe":            round(sharpe, 3),
        "Max Drawdown %":    round(max_dd, 2),
        "Win Rate %":        round(win_rate, 2),
    }
    print(f"\n── {namn} ──")
    for k, v in stats.items():
        if k != "Namn":
            print(f"  {k}: {v}")
    return stats


# ─────────────────────────────────────────
# 7. VISUALISERING
# ─────────────────────────────────────────
def visa_resultat(bot_portfölj: pd.Series, index_bah: pd.Series):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bot_portfölj.index, y=bot_portfölj.values,
        mode="lines", name="Modell 2 (LLM-sentiment)",
        line=dict(color="steelblue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=index_bah.index, y=index_bah.values,
        mode="lines", name="OMXS30 Buy & Hold",
        line=dict(dash="dash", color="black", width=2)
    ))

    fig.update_layout(
        title="Modell 2: Nyhetsbaserad LLM-bot vs OMXS30 Buy & Hold",
        xaxis_title="Datum",
        yaxis_title="Portföljvärde (SEK)",
        template="plotly_white",
        legend=dict(x=0, y=1),
    )
    fig.show()


# ─────────────────────────────────────────
# 8. HUVUDPROGRAM
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Modell 2 – Nyhetsbaserad riskanalys med LLM")
    print("=" * 55)

    if LLM_BACKEND == "groq" and "DIN_GROQ_NYCKEL_HÄR" in LLAMA_API_KEY:
        print("\nOBS: Sätt din API-nyckel i variabeln LLAMA_API_KEY.")
        print("     Skapa ett gratis konto på https://console.groq.com")
        print("     Eller sätt LLM_BACKEND = 'ollama' för helt lokal körning.\n")

    # --- Hämta aktiedata ---
    print(f"Hämtar prisdata {TEST_START} → {TEST_END}...")
    raw = yf.download(OMXS30_TICKERS, start=TEST_START, end=TEST_END,
                      interval="1d",  # daglig upplösning per rapporten
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        pris_data = raw["Close"]
    else:
        pris_data = raw[["Close"]]
        pris_data.columns = OMXS30_TICKERS

    pris_data = pris_data.sort_index().dropna(how="all")

    # Index för buy & hold
    index_raw  = yf.download(INDEX_TICKER, start=TEST_START, end=TEST_END,
                              interval="1d", auto_adjust=True, progress=False)
    index_pris = index_raw["Close"].squeeze()
    index_bah  = buy_and_hold(index_pris, INITIAL_CAPITAL)

    # --- Hämta nyheter & analysera med Llama ---
    nyheter = hamta_alla_nyheter()

    if not nyheter:
        print("Inga nyheter hämtades. Kontrollera nätverksanslutning.")
        return

    # Kör LLM-analys för varje aktie baserat på dagens nyheter
    # (i backtesting-läge: samma nyheter appliceras på senaste tillgängliga dag)
    dag       = pris_data.index[-1]
    signaler  = pd.DataFrame(0, index=[dag], columns=OMXS30_TICKERS)

    print(f"\nAnalyserar nyheter med Llama för {dag.date()}...")
    for ticker in OMXS30_TICKERS:
        if ticker not in pris_data.columns:
            continue
        if pd.isna(pris_data.loc[dag, ticker]):
            continue

        aktuellt_pris = float(pris_data.loc[dag, ticker])
        analyser = []

        for artikel in nyheter:
            print(f"  [{ticker}] {artikel['kalla']}: {artikel['titel'][:60]}...")
            analys = analysera_med_llama(artikel, aktuellt_pris, ticker)
            if analys:
                analyser.append(analys)
                print(f"    → sentiment={analys['sentiment']}, ΔP={analys['delta_p']:+.2f} SEK")
            time.sleep(0.3)  # undvik rate limiting

        signal, förväntat = generera_signal(analyser, aktuellt_pris)
        signaler.loc[dag, ticker] = signal
        print(f"  [{ticker}] Signal: {signal}, förväntat pris: {förväntat:.2f} SEK")

    # --- Simulera handel ---
    bot_portfölj = simulera_handel(signaler, pris_data, INITIAL_CAPITAL)

    # --- Statistik ---
    berakna_statistik(bot_portfölj, "Modell 2 (LLM-sentiment)")
    berakna_statistik(index_bah,    "OMXS30 Buy & Hold")

    # --- Visualisering ---
    visa_resultat(bot_portfölj, index_bah)

    print("\nKlart!")


if __name__ == "__main__":
    main()
