import io
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

# =====================================================
# CONFIG
# =====================================================

LOOKBACK_TD = 60      # trading days (Yahoo)
LOOKBACK_DAYS = 90    # calendar days (FRED); comparable to ~60 trading days
MIN_COVERAGE = 0.60   # below this, the output is flagged as unreliable
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

YAHOO_SYMBOLS = {
    "copper": "HG=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "hyg": "HYG",
    "lqd": "LQD",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "dbc": "DBC",
}

FRED_SERIES = {
    "sofr": "SOFR",
    "dgs10": "DGS10",     # daily (replaces monthly GS10)
    "dgs2": "DGS2",       # daily (replaces monthly GS2)
    "nfci": "NFCI",       # weekly
    "unrate": "UNRATE",   # monthly
    "dfii10": "DFII10",   # daily 10y real yield
    "t10yie": "T10YIE",   # daily 10y breakeven
}

failed = []


# =====================================================
# DATA LAYER
# =====================================================

def pct_change(now, old):
    if now is None or old is None or old == 0:
        return None
    return (now / old - 1) * 100


def ratio_change(a, b):
    """% change in the ratio a/b. a and b are (now, old) tuples."""
    a_now, a_old = a
    b_now, b_old = b
    if None in (a_now, a_old, b_now, b_old) or b_now == 0 or b_old == 0 or a_old == 0:
        return None
    return ((a_now / b_now) / (a_old / b_old) - 1) * 100


def yahoo_close(symbol):
    """(latest, value LOOKBACK_TD trading days ago)"""
    try:
        df = yf.download(symbol, period="1y", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("empty download")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) <= LOOKBACK_TD:
            raise ValueError(f"only {len(close)} bars")
        return float(close.iloc[-1]), float(close.iloc[-LOOKBACK_TD - 1])
    except Exception as e:
        failed.append(symbol)
        print(f"[WARN] yahoo {symbol}: {e}")
        return None, None


def fred_latest(series, days=LOOKBACK_DAYS):
    """(latest, last observation at least `days` calendar days before latest)"""
    try:
        r = requests.get(FRED_CSV.format(series), timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), na_values=["."], index_col=0, parse_dates=True)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        if s.empty:
            raise ValueError("empty series")
        cutoff = s.index[-1] - pd.Timedelta(days=days)
        prior = s[s.index <= cutoff]
        old = float(prior.iloc[-1]) if not prior.empty else None
        return float(s.iloc[-1]), old
    except Exception as e:
        failed.append(series)
        print(f"[WARN] fred {series}: {e}")
        return None, None


# =====================================================
# SCORING TABLE (current regime only)
# =====================================================
# fn(value) -> +1 growth, -1 defensive, 0 neutral
# w_cur: weight in current regime

@dataclass
class Rule:
    name: str
    key: str
    fmt: str
    fn: Callable[[float], int]
    text: dict            # {1: ..., -1: ..., 0: ...}
    w_cur: int


def band(hi, lo, invert=False):
    def f(v):
        s = 1 if v > hi else -1 if v < lo else 0
        return -s if invert else s
    return f


RULES = [
    Rule("Copper/Gold (60d)", "copper_gold", "{:+.2f}%", band(5, -5),
         {1: "Bullish (growth demand)", -1: "Bearish (defensive)", 0: "Neutral"}, 3),
    Rule("HYG/LQD credit (60d)", "credit", "{:+.2f}%", band(2, -2),
         {1: "Credit risk-on", -1: "Credit risk-off", 0: "Neutral"}, 3),
    Rule("Yield curve 10y-2y", "curve", "{:+.2f}", band(0.50, 0.0),
         {1: "Positive curve", -1: "Inverted curve", 0: "Flat (0 to 0.5)"}, 3),
    Rule("Gold/Silver (60d)", "gold_silver", "{:+.2f}%", band(3, -3, invert=True),
         {1: "Silver outperforming", -1: "Gold outperforming", 0: "Neutral"}, 1),
    Rule("VIX", "vix", "{:.2f}", band(25, 18, invert=True),
         {1: "Low volatility", -1: "Elevated volatility", 0: "Neutral"}, 2),
    Rule("Commodities DBC (60d)", "dbc", "{:+.2f}%", band(5, -5),
         {1: "Commodity demand strong", -1: "Commodity demand weak", 0: "Neutral"}, 1),
    Rule("Dollar DXY (60d)", "dxy", "{:+.2f}%", band(3, -3, invert=True),
         {1: "Dollar weakening (looser global liquidity)", -1: "Dollar strengthening (tighter)", 0: "Neutral"}, 1),
    Rule("NFCI", "nfci", "{:+.2f}", band(0.0, 0.0, invert=True),
         {1: "Easy financial conditions", -1: "Tight financial conditions", 0: "Neutral"}, 2),
    Rule("Unemployment chg (90d)", "unrate_chg", "{:+.2f}pp", band(0.1, -0.1, invert=True),
         {1: "Labor market improving", -1: "Labor market weakening", 0: "Stable"}, 2),
    Rule("SOFR chg (90d)", "sofr_chg", "{:+.2f}pp", band(0.10, -0.10, invert=True),
         {1: "Policy rates falling (easing)", -1: "Policy rates rising (restrictive)", 0: "Unchanged"}, 1),
    Rule("10y real yield (TIPS)", "real10", "{:.2f}%", band(2.0, 2.0, invert=True),
         {1: "Real yields <= 2% (not restrictive)", -1: "Real yields > 2% (restrictive)", 0: "Neutral"}, 1),
]

RULE_WEIGHTS = {r.name: r.w_cur for r in RULES}


def build_metrics(y, f):
    return {
        "copper_gold": ratio_change(y["copper"], y["gold"]),
        "credit": ratio_change(y["hyg"], y["lqd"]),
        "curve": round(f["dgs10"][0] - f["dgs2"][0], 4) if f["dgs10"][0] is not None and f["dgs2"][0] is not None else None,
        "gold_silver": ratio_change(y["gold"], y["silver"]),
        "vix": y["vix"][0],
        "dbc": pct_change(*y["dbc"]),
        "dxy": pct_change(*y["dxy"]),
        "nfci": f["nfci"][0],
        "unrate_chg": round(f["unrate"][0] - f["unrate"][1], 4) if None not in f["unrate"] else None,
        "sofr_chg": round(f["sofr"][0] - f["sofr"][1], 4) if None not in f["sofr"] else None,
        "real10": f["dfii10"][0],
    }


LADDER = [
    (0.60, "EXPANSION"),
    (0.30, "RECOVERY"),
    (0.00, "SOFT LANDING"),
    (-0.30, "EARLY SLOWDOWN"),
    (-0.60, "RECESSION WARNING"),
]


def classify(ratio):
    for threshold, name in LADDER:
        if ratio >= threshold:
            return name
    return "RECESSION"


# =====================================================
# STANCE & DIAGNOSTIC HELPERS
# =====================================================

STANCE = {
    "EXPANSION": (
        "NORMAL RISK",
        "Growth is broad and stress indicators are low.",
    ),
    "RECOVERY": (
        "NORMAL RISK",
        "Most indicators remain supportive.",
    ),
    "SOFT LANDING": (
        "NORMAL RISK",
        "Mixed signals but no broad stress cluster.",
    ),
    "EARLY SLOWDOWN": (
        "NORMAL RISK / NOT A SELL",
        "Growth is cooling but stress is not widespread.",
    ),
    "RECESSION WARNING": (
        "DEFENSIVE",
        "Stress is broad across markets and macro indicators.",
    ),
    "RECESSION": (
        "CAPITULATION WATCH",
        "Extreme stress environment. Avoid emotional decisions.",
    ),
}


def stance(regime):
    return STANCE[regime]


def score(metrics):
    rows = []
    cur = 0
    cur_max = 0
    growth = 0
    defensive = 0
    for r in RULES:
        v = metrics.get(r.key)
        if v is None:
            rows.append((r.name, "n/a", "MISSING DATA", 0))
            continue
        s = r.fn(v)
        cur += s * r.w_cur
        cur_max += r.w_cur
        if s > 0:
            growth += r.w_cur
        elif s < 0:
            defensive += r.w_cur
        rows.append(
            (
                r.name,
                r.fmt.format(v),
                r.text[s],
                s * r.w_cur,
            )
        )
    cur_ratio = cur / cur_max if cur_max else 0.0
    loaded = sum(1 for r in rows if r[1] != "n/a")
    coverage = loaded / len(RULES)
    return {
        "rows": rows,
        "growth": growth,
        "defensive": defensive,
        "loaded_weight": cur_max,
        "cur_ratio": cur_ratio,
        "coverage": coverage,
        "loaded": loaded,
    }


def strongest_signals(rows, n=3):
    valid = [r for r in rows if r[1] != "n/a"]
    return sorted(
        valid,
        key=lambda x: (abs(x[3]), RULE_WEIGHTS.get(x[0], 0)),
        reverse=True,
    )[:n]


def market_stress(ratio):
    mag = abs(ratio)
    if mag <= 0.20:
        return "LOW"
    if mag <= 0.40:
        return "MODERATE"
    if mag <= 0.60:
        return "HIGH"
    return "EXTREME"


def vix_risk(vix):
    if vix is None:
        return "UNKNOWN"
    if vix < 18:
        return "LOW"
    if vix < 25:
        return "NORMAL"
    if vix < 35:
        return "HIGH"
    return "EXTREME"


def breadth(res):
    """Share of loaded weight flashing growth / defensive / neutral, plus dominant signal."""
    w = res["loaded_weight"]
    if not w:
        return 0.0, 0.0, 0.0, 0.0, "Balanced"
    g = res["growth"] / w * 100
    d = res["defensive"] / w * 100
    n = 100 - g - d
    dominant_pct = max(g, d)
    dominant_side = (
        "Growth"
        if g > d
        else "Defensive"
        if d > g
        else "Balanced"
    )
    return g, d, n, dominant_pct, dominant_side


# =====================================================
# MAIN
# =====================================================

def main():
    print("\nDownloading data...\n")
    y = {k: yahoo_close(sym) for k, sym in YAHOO_SYMBOLS.items()}
    f = {k: fred_latest(sid) for k, sid in FRED_SERIES.items()}

    metrics = build_metrics(y, f)
    res = score(metrics)

    breakeven = f["t10yie"][0]
    regime = classify(res["cur_ratio"])
    g, d, n, dom_pct, dom_side = breadth(res)
    st, note = stance(regime)

    print("=" * 80)
    print("INSTITUTIONAL MACRO DASHBOARD")
    print("=" * 80)
    print(f"\nDate: {datetime.now():%Y-%m-%d}")

    if res["coverage"] < MIN_COVERAGE:
        print(
            f"\n*** LOW DATA COVERAGE: {res['loaded']}/{len(RULES)} inputs loaded. "
            f"Treat results as unreliable. ***"
        )

    print("\nCURRENT REGIME")
    print(f"Regime          : {regime}")
    print(f"Net score       : {res['cur_ratio']:+.2f}  (range -1 to +1)")
    print(f"Stance          : {st}")
    print(f"Read as         : {note}")

    print("\nSIGNAL BREADTH")
    print(f"Growth          : {g:5.1f}%")
    print(f"Defensive       : {d:5.1f}%")
    print(f"Neutral         : {n:5.1f}%")
    print(f"Dominant Signal : {dom_pct:5.1f}% ({dom_side})")
    print(f"Inputs loaded   : {res['loaded']}/{len(RULES)}")

    print("\nMARKET STRESS")
    print(f"Stress Level : {market_stress(res['cur_ratio'])}")

    print("\nVOLATILITY")
    if metrics["vix"] is not None:
        print(f"VIX          : {metrics['vix']:.2f}")
        print(f"Risk State   : {vix_risk(metrics['vix'])}")
    else:
        print("VIX          : n/a")

    print("\nTOP DRIVERS")
    for name, val, signal, pts in strongest_signals(res["rows"]):
        print(f"• {name} ({pts:+d})  {val} — {signal}")

    print("\nSIGNALS")
    print(f"{'Input':<26}{'Value':>12}  {'Signal':<46}{'Pts':>4}")
    for name, val, text, pts in res["rows"]:
        print(f"{name:<26}{val:>12}  {text:<46}{pts:+4}")

    print("\nINFO")
    if breakeven is not None:
        print(f"• 10y Breakeven (T10YIE): {breakeven:.2f}%")
    if failed:
        print(f"• Failed downloads: {', '.join(failed)}")
    print("• Dashboard measures macro and market stress.")
    print("• Designed as a situational-awareness tool, not a return predictor.")
    print("• Regime labels describe conditions, not expected market direction.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()