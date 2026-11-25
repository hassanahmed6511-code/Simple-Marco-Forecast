#!/usr/bin/env python3
"""
Backtest of the macro regime dashboard against US equities.

Imports RULES / score / classify from your dashboard file, so the logic tested
is exactly the logic you run live. Rebuilds every input point-in-time:

- Yahoo series : same-day close, LOOKBACK_TD-bar change (as in yahoo_close)
- FRED series  : value known on date t = last obs with obs_date + publication lag <= t
                   90-day change = obs vs last obs >= LOOKBACK_DAYS earlier (as in fred_latest)
- Signal at close t -> position earns the return of t+exec_lag (no look-ahead)

Usage:
    python backtest_macro.py --src macro_dashboard.py
    python backtest_macro.py --src macro_dashboard.py --start 2008-01-01 --rebal M --cost-bps 10
"""
import argparse
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(".bt_cache")
HIST_START = "1990-01-01"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# Publication lag (calendar days) before an observation is usable.
FRED_LAG = {
    "dgs10": 1, "dgs2": 1, "dfii10": 1, "t10yie": 1, "sofr": 1,
    "nfci": 5,      # weekly, released Wednesday for week ending prior Friday
    "unrate": 35,   # obs dated 1st of month, released ~first Friday of next month
}

# Equity exposure per regime (remainder in T-bills). These are MY assumptions - edit freely.
ALLOC = {
    "EXPANSION": 1.00,
    "RECOVERY": 1.00,
    "SOFT LANDING": 0.80,
    "EARLY SLOWDOWN": 0.50,
    "RECESSION WARNING": 0.25,
    "RECESSION": 0.00,
}


# =====================================================
# DATA
# =====================================================

def _empty():
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _cached(name, refresh, fetch):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.pkl"
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    s = fetch()
    pd.to_pickle(s, p)
    return s


def get_yahoo(sym, refresh=False):
    def fetch():
        import yfinance as yf
        df = yf.download(sym, start=HIST_START, auto_adjust=True, progress=False)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.dropna().astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s[~s.index.duplicated()].sort_index()
    try:
        return _cached("y_" + "".join(c for c in sym if c.isalnum()), refresh, fetch)
    except Exception as e:
        print(f"[WARN] yahoo {sym}: {e}")
        return _empty()


def get_fred(sid, refresh=False):
    def fetch():
        r = requests.get(FRED_CSV.format(sid), timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), na_values=["."], index_col=0, parse_dates=True)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return s[~s.index.duplicated()].sort_index()
    try:
        return _cached("f_" + sid, refresh, fetch)
    except Exception as e:
        print(f"[WARN] fred {sid}: {e}")
        return _empty()


# =====================================================
# POINT-IN-TIME FEATURES
# =====================================================

def on_cal(s, cal):
    """Forward-fill a series onto the trading calendar."""
    return s.reindex(s.index.union(cal)).ffill().reindex(cal)


def pit(s, cal, lag_days):
    """Value known on each calendar date: obs dated d becomes visible at d + lag."""
    a = s.copy()
    a.index = a.index + pd.Timedelta(days=lag_days)
    return on_cal(a, cal)


def chg_vs_lookback(s, days):
    """Per observation: value minus last obs at least `days` calendar days earlier (fred_latest logic)."""
    if s.empty:
        return s
    cutoff = s.index - pd.Timedelta(days=days)
    pos = s.index.searchsorted(cutoff, side="right") - 1
    old = np.where(pos >= 0, s.values[np.clip(pos, 0, None)], np.nan)
    return s - pd.Series(old, index=s.index)


def build_metrics(mod, cal, Y, F):
    LB, DAYS = mod.LOOKBACK_TD, mod.LOOKBACK_DAYS
    y = {k: on_cal(v, cal) for k, v in Y.items()}

    def ch(s):
        return (s / s.shift(LB) - 1) * 100

    def rc(a, b):
        return ((a / b) / (a.shift(LB) / b.shift(LB)) - 1) * 100

    f = {k: pit(v, cal, FRED_LAG[k]) for k, v in F.items()}
    m = pd.DataFrame(index=cal)
    m["copper_gold"] = rc(y["copper"], y["gold"])
    m["credit"] = rc(y["hyg"], y["lqd"])
    m["curve"] = (f["dgs10"] - f["dgs2"]).round(4)
    m["gold_silver"] = rc(y["gold"], y["silver"])
    m["vix"] = y["vix"]
    m["dbc"] = ch(y["dbc"])
    m["dxy"] = ch(y["dxy"])
    m["nfci"] = f["nfci"]
    m["unrate_chg"] = pit(chg_vs_lookback(F["unrate"], DAYS), cal, FRED_LAG["unrate"]).round(4)
    m["sofr_chg"] = pit(chg_vs_lookback(F["sofr"], DAYS), cal, FRED_LAG["sofr"]).round(4)
    m["real10"] = f["dfii10"]
    return m.replace([np.inf, -np.inf], np.nan)


def run_dashboard(mod, m):
    rows = []
    for _, row in m.iterrows():
        met = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        r = mod.score(met)
        rows.append((
            r["cur_ratio"],
            r["coverage"],
            r["growth"],
            r["defensive"],
        ))

    sig = pd.DataFrame(
        rows,
        index=m.index,
        columns=["cur_ratio", "coverage", "growth", "defensive"],
    )
    sig["cur_reg"] = [mod.classify(x) for x in sig["cur_ratio"]]
    return sig


# =====================================================
# STRATEGY + STATS
# =====================================================

def weights(regimes, rebal, smooth=1):
    w = regimes.map(ALLOC).astype(float)
    if smooth > 1:
        w = w.rolling(smooth, min_periods=1).mean()
    if rebal != "D":
        per = w.index.to_period(rebal)
        is_last = w.index.to_series().groupby(per).transform("max").values == w.index.values
        w = w.where(is_last).ffill()
    return w


def strat_returns(w, ret, rf, lag, cost_bps):
    pos = w.shift(lag)
    turn = pos.diff().abs().fillna(0)
    r = pos * ret + (1 - pos) * rf - turn * cost_bps / 1e4
    return r.dropna(), pos.reindex(r.dropna().index), turn.reindex(r.dropna().index)


def stats(r, rf, pos=None, turn=None):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    ex = r - rf.reindex(r.index).fillna(0)
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    out = {
        "CAGR%": cagr * 100,
        "Vol%": vol * 100,
        "Sharpe": sharpe,
        "MaxDD%": mdd * 100,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "AvgExp%": pos.mean() * 100 if pos is not None else 100.0,
        "Turnover/yr": turn.sum() / (n / 252) if turn is not None else 0.0,
    }
    return out


def regime_table(df, col, ladder_names):
    g = df.groupby(col)
    t = pd.DataFrame({
        "days%": g.size() / len(df) * 100,
        "fwd21_mean%": g["fwd21"].mean() * 100,
        "fwd21_hit%": g["fwd21"].apply(lambda x: (x.dropna() > 0).mean() * 100),
        "fwd63_mean%": g["fwd63"].mean() * 100,
        "fwd63_worst%": g["fwd63"].min() * 100,
        "nber_rec%": g["rec"].mean() * 100,
    })
    return t.reindex([n for n in ladder_names if n in t.index])


# =====================================================
# MAIN
# =====================================================

def apply_exposure(mod, legacy):
    """Use the dashboard's EXPOSURE map unless --legacy-alloc is given."""
    if hasattr(mod, "EXPOSURE") and not legacy:
        ALLOC.update(mod.EXPOSURE)
    print("Allocation map:", {k: f"{v:.0%}" for k, v in ALLOC.items()})


def load_dashboard(path):
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent / path
    if not p.exists():
        sys.exit(f"dashboard file not found: {path}")
    spec = importlib.util.spec_from_file_location("dashboard", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="macro_dashboard.py", help="path to your dashboard file")
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--bench", default="SPY", help="total-return proxy (auto-adjusted)")
    ap.add_argument("--rebal", default="W", choices=["D", "W", "M"])
    ap.add_argument("--exec-lag", type=int, default=1, help="trading days between signal and position")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="per unit of turnover")
    ap.add_argument("--sofr", default="DFF",
                    help="series for the sofr slot. SOFR only starts 2018; DFF (fed funds) spans the sample")
    ap.add_argument("--refresh", action="store_true", help="ignore cache")
    ap.add_argument("--smooth", type=int, default=1,
                    help="average target weights over N trading days (21 ~ monthly tranches). Use with --rebal D")
    ap.add_argument("--offsets", action="store_true",
                    help="rebalance-day robustness: rerun with the monthly rebalance on each of trading days 1-21")
    ap.add_argument("--legacy-alloc", action="store_true",
                    help="use the original 100/100/80/50/25/0 map instead of the dashboard's EXPOSURE")
    ap.add_argument("--out", default="backtest_out")
    a = ap.parse_args(argv)

    mod = load_dashboard(a.src)
    apply_exposure(mod, a.legacy_alloc)
    Path(a.out).mkdir(exist_ok=True)

    bench = get_yahoo(a.bench, a.refresh)
    if bench.empty:
        sys.exit(f"benchmark {a.bench} download failed")
    cal = bench.index

    print("Loading data...")
    Y = {k: get_yahoo(s, a.refresh) for k, s in mod.YAHOO_SYMBOLS.items()}
    F = {k: get_fred(a.sofr if k == "sofr" else sid, a.refresh) for k, sid in mod.FRED_SERIES.items()}
    rf_d = pit(get_fred("DTB3", a.refresh), cal, 1).ffill().fillna(0) / 100 / 252
    rec = on_cal(get_fred("USREC", a.refresh), cal).fillna(0)

    print("Scoring history...")
    m = build_metrics(mod, cal, Y, F)
    sig = run_dashboard(mod, m)

    ret = bench.pct_change()
    df = sig.copy()
    df["ret"], df["rf"], df["rec"] = ret, rf_d, rec
    df["fwd21"] = bench.shift(-21) / bench - 1
    df["fwd63"] = bench.shift(-63) / bench - 1

    sample = (df.index >= pd.Timestamp(a.start)) & (df["coverage"] >= mod.MIN_COVERAGE)
    d = df[sample].copy()
    if d.empty:
        sys.exit("no days with sufficient coverage - check downloads / --start")
    print(f"\nSample: {d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d}  ({len(d)} days)"
          f"  | coverage>={mod.MIN_COVERAGE:.0%} | benchmark {a.bench}")
    print("Avg input coverage by year:")
    print((sig.loc[sig.index >= pd.Timestamp(a.start), "coverage"]
           .groupby(lambda x: x.year).mean() * 100).round(0).astype(int).to_string())

    names = [n for _, n in mod.LADDER] + ["RECESSION"]
    pd.set_option("display.width", 200)
    print(f"\n=== Forward {a.bench} returns by CURRENT regime (overlapping windows; n is effectively smaller) ===")
    print(regime_table(d, "cur_reg", names).round(2).to_string())
    base = d[["fwd21", "fwd63"]].mean() * 100
    print(f"\nUnconditional: fwd21 {base['fwd21']:.2f}%  fwd63 {base['fwd63']:.2f}%")

    print(f"\n=== Strategies (rebal={a.rebal}, smooth={a.smooth}d, exec lag={a.exec_lag}d, cost={a.cost_bps}bps, cash=3m T-bill) ===")
    results, curves = {}, {}
    bh = d["ret"].dropna()
    results["Buy&Hold"] = stats(bh, d["rf"])
    curves["Buy&Hold"] = bh

    for label, col in (("Current", "cur_reg"),):
        w = weights(d[col], a.rebal, a.smooth)
        r, pos, turn = strat_returns(w, d["ret"], d["rf"], a.exec_lag, a.cost_bps)
        results[label] = stats(r, d["rf"], pos, turn)
        curves[label] = r
        d[f"{label}_w"] = pos

    avg_w = d["Current_w"].mean()
    d["Static_w"] = avg_w
    curves["Static"] = avg_w * d["ret"] + (1 - avg_w) * d["rf"]
    print(f"(Static = constant {avg_w:.0%} equity / {1 - avg_w:.0%} T-bills, same average exposure as Current)")

    common = pd.concat(curves, axis=1).dropna()
    results = {
        k: stats(
            common[k],
            d["rf"],
            d[f"{k}_w"].reindex(common.index) if k != "Buy&Hold" else None,
            d[f"{k}_w"].reindex(common.index).diff().abs().fillna(0) if k != "Buy&Hold" else None,
        )
        for k in common.columns
    }
    print(pd.DataFrame(results).T.round(2).to_string())

    yr = (1 + common).groupby(common.index.year).prod() - 1
    print("\n=== Calendar-year returns (%) ===")
    print((yr * 100).round(1).to_string())

    if a.offsets:
        print("\n=== Rebalance-day robustness (monthly rebalance on trading day k of the month) ===")
        rank = pd.Series(range(len(d)), index=d.index).groupby(d.index.to_period("M")).cumcount().values
        rows = []
        for k in range(21):
            row = {"offset": k}
            base = d["cur_reg"].map(ALLOC).astype(float)
            if a.smooth > 1:
                base = base.rolling(a.smooth, min_periods=1).mean()
            w = base.where(rank == k).ffill()
            r, pos, turn = strat_returns(w, d["ret"], d["rf"], a.exec_lag, a.cost_bps)
            st = stats(r, d["rf"], pos, turn)
            row["Cur_CAGR%"], row["Cur_Sharpe"], row["Cur_MaxDD%"] = st["CAGR%"], st["Sharpe"], st["MaxDD%"]
            rows.append(row)
        t = pd.DataFrame(rows).set_index("offset")
        print(t.round(2).to_string())
        print("\nSpread across offsets:")
        print(t.agg(["min", "median", "max"]).round(2).to_string())
        bh_s = results["Buy&Hold"]["Sharpe"]
        print(f"\nBuy&Hold Sharpe {bh_s:.2f}. Offsets beating it: Cur {(t['Cur_Sharpe'] > bh_s).sum()}/21")

    d.to_csv(Path(a.out) / "daily_signals.csv")
    common.to_csv(Path(a.out) / "strategy_returns.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eq = (1 + common).cumprod()
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1])
        for c in eq.columns:
            ax[0].plot(eq.index, eq[c], label=c, lw=2 if c == "Buy&Hold" else 1.3)
            ax[1].plot(eq.index, eq[c] / eq[c].cummax() - 1, lw=1)
        ax[0].set_yscale("log")
        ax[0].legend()
        ax[0].set_title("Macro regime backtest (growth of $1)")
        ax[1].set_title("Drawdown")
        fig.tight_layout()
        fig.savefig(Path(a.out) / "equity_curves.png", dpi=130)
        print(f"\nSaved: {a.out}/daily_signals.csv, strategy_returns.csv, equity_curves.png")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")


if __name__ == "__main__":
    main()