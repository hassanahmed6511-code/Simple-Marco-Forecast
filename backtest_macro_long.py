#!/usr/bin/env python3
"""
Long-history (1920+) monthly backtest of the macro regime dashboard.

Needs backtest_macro.py in the same folder (reuses its loaders / point-in-time helpers).

Most of the dashboard's inputs don't exist before 1971 (HYG 2007, VIX 1990, DXY 1971, NFCI 1971,
copper/gold futures 2000, 2y yield 1976, SOFR 2018). Where the real input is missing this script
substitutes a proxy; where no proxy exists the input is left missing and score() renormalises
over whatever is left. Early-era results therefore come from 3-6 inputs, not 11.

  input         real source (as in your script)      proxy used when real one doesn't exist
  ------------  -----------------------------------  ----------------------------------------------
  credit        HYG/LQD 60d % chg (2007+)            -K * 3m chg in Baa-Aaa yield spread (1919+)
  curve         10y-2y                               10y-3m (GS10 - TB3MS, 1934+; Shiller GS10 pre-1953)
  vix           ^VIX (1990+)                         21d realised vol x RV_TO_VIX (needs daily ^GSPC, 1928+)
  dbc           DBC 60d % chg (2006+)                PPI all-commodities 3m % chg (1913+)
  real10        DFII10 (2003+)                       GS10 - trailing 12m CPI inflation (1914+)
  sofr_chg      DFF 90d chg (1954+)                  TB3MS 3m chg (1934+)
  unrate_chg    UNRATE (1948+)                       none
  nfci          NFCI (1971+)                         none
  dxy           DX-Y.NYB (1971+)                     none
  copper/gold,
  gold/silver   futures (2000+)                      none

Equity: ^GSPC month-end close from 1928; Shiller monthly series before that (spliced; its prices are
monthly averages, so 1920-27 returns are smoothed). Dividends added from Shiller yield. Price-only if
Shiller is unavailable. Cash: 3m T-bill (TB3MS); before 1934 approximated as Shiller GS10 minus 1pp.

Usage:
    python backtest_macro_long.py --src macro_dashboard.py
    python backtest_macro_long.py --src macro_dashboard.py --shiller ie_data.xls --min-cov 0.35
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_macro as bm

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
FRED_LONG = ["AAA", "BAA", "GS10", "TB3MS", "CPIAUCNS", "PPIACO", "USREC", "DTB3"]
LAG = {"AAA": 35, "BAA": 35, "GS10": 35, "TB3MS": 35, "CPIAUCNS": 45, "PPIACO": 45}

CREDIT_K = 4.0     # proxy: credit% = -K * d3m(Baa-Aaa, pp)  => +-2% threshold ~ +-0.5pp spread move
RV_TO_VIX = 1.25   # VIX has historically run ~1.25x realised vol

ERAS = [("1920-1947", "1920-01-01", "1947-12-31"), ("1948-1970", "1948-01-01", "1970-12-31"),
        ("1971-2006", "1971-01-01", "2006-12-31"), ("2007-now", "2007-01-01", "2100-01-01")]
CRISES = [("1929-32 crash", "1929-09-30", "1932-06-30"), ("1937-38", "1937-02-28", "1938-03-31"),
          ("1973-74", "1973-01-31", "1974-09-30"), ("1987", "1987-08-31", "1987-12-31"),
          ("2000-02", "2000-03-31", "2002-09-30"), ("2008-09", "2007-10-31", "2009-02-28"),
          ("2020 Covid", "2020-01-31", "2020-03-31"), ("2022", "2021-12-31", "2022-09-30")]


# =====================================================
# HELPERS
# =====================================================

def to_me(x):
    """Last value per calendar month, labelled with the month-end date (works for Series/DataFrame)."""
    g = x.groupby(x.index.to_period("M")).last()
    g.index = g.index.to_timestamp(how="end").normalize()
    return g


def get_shiller(path, refresh):
    def fetch():
        raw = pd.read_excel(path or SHILLER_URL, sheet_name="Data", header=None).iloc[:, :7]
        raw.columns = ["date", "P", "D", "E", "CPI", "frac", "GS10"]
        raw["date"] = pd.to_numeric(raw["date"], errors="coerce")
        raw = raw[(raw["date"] >= 1871) & (raw["date"] < 2100)].copy()
        for c in ["P", "D", "CPI", "GS10"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        yr = raw["date"].astype(int)
        mo = ((raw["date"] - yr) * 100).round().astype(int)   # 1920.1 -> October
        idx = pd.to_datetime(dict(year=yr, month=mo, day=1)) + pd.offsets.MonthEnd(0)
        raw.index = pd.DatetimeIndex(idx)
        return raw[["P", "D", "CPI", "GS10"]]
    try:
        return bm._cached("shiller", refresh, fetch)
    except Exception as e:
        print(f"[WARN] Shiller data unavailable ({e}). Falling back to ^GSPC only (starts 1928). "
              f"Download ie_data.xls from Shiller's site and pass --shiller PATH.")
        return None


def build_equity(gspc, sh):
    me = to_me(gspc)
    me = me[me.index >= pd.Timestamp("1928-01-31")]
    dy = pd.Series(0.0, index=me.index)
    px = me
    if sh is not None:
        shp = sh["P"].dropna()
        ref = shp.reindex([me.index[0]]).iloc[0]
        if pd.notna(ref):
            pre = shp[shp.index < me.index[0]] * (me.iloc[0] / ref)
            px = pd.concat([pre, me])
        dy = (sh["D"] / sh["P"] / 12).reindex(px.index).ffill().fillna(0.0)
    else:
        print("[WARN] no dividend data: returns are price-only")
    px = px[~px.index.duplicated()].sort_index()
    ret = px.pct_change() + dy
    return px, ret


# =====================================================
# METRICS (real where they exist, proxies elsewhere)
# =====================================================

def build_long_metrics(mod, cal, gspc, Y, F, sh):
    # 1) real inputs, computed exactly as in the daily backtest, then sampled at month-end
    m_d = bm.build_metrics(mod, gspc.index, Y, {k: v for k, v in F.items() if k in mod.FRED_SERIES})
    m = to_me(m_d).reindex(cal)

    # 2) proxies on the monthly calendar
    P = lambda s, lag: bm.pit(s, cal, lag)
    aaa, baa = P(F["AAA"], LAG["AAA"]), P(F["BAA"], LAG["BAA"])
    spread = baa - aaa
    gs10 = P(F["GS10"], LAG["GS10"])
    if sh is not None:
        gs10 = gs10.combine_first(P(sh["GS10"].dropna(), LAG["GS10"]))
    tb3 = P(F["TB3MS"], LAG["TB3MS"])
    cpi = P(F["CPIAUCNS"], LAG["CPIAUCNS"])
    ppi = P(F["PPIACO"], LAG["PPIACO"])
    rv = np.log(gspc).diff().rolling(21).std() * np.sqrt(252) * 100 * RV_TO_VIX

    proxy = {
        "credit": -CREDIT_K * (spread - spread.shift(3)),
        "curve": gs10 - tb3,
        "vix": to_me(rv).reindex(cal),
        "dbc": (ppi / ppi.shift(3) - 1) * 100,
        "real10": gs10 - (cpi / cpi.shift(12) - 1) * 100,
        "sofr_chg": tb3 - tb3.shift(3),
    }
    src = pd.DataFrame(index=cal)
    for k, p in proxy.items():
        src[k] = np.where(m[k].notna(), "real", np.where(p.notna(), "proxy", "none"))
        m[k] = m[k].fillna(p)
    return m.replace([np.inf, -np.inf], np.nan), src


# =====================================================
# STATS
# =====================================================

def stats(r, rf, pos=None, turn=None, ppy=12):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (ppy / n) - 1
    ex = r - rf.reindex(r.index).fillna(0)
    mdd = (eq / eq.cummax() - 1).min()
    return {"CAGR%": cagr * 100, "Vol%": r.std() * np.sqrt(ppy) * 100,
            "Sharpe": ex.mean() / ex.std() * np.sqrt(ppy) if ex.std() > 0 else np.nan,
            "MaxDD%": mdd * 100, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "AvgExp%": pos.mean() * 100 if pos is not None else 100.0,
            "Turnover/yr": turn.sum() / (n / ppy) if turn is not None else 0.0}


def regime_table(df, col, names):
    g = df.groupby(col)
    t = pd.DataFrame({
        "months%": g.size() / len(df) * 100,
        "fwd3m_mean%": g["fwd3"].mean() * 100,
        "fwd3m_hit%": g["fwd3"].apply(lambda x: (x.dropna() > 0).mean() * 100),
        "fwd12m_mean%": g["fwd12"].mean() * 100,
        "fwd12m_worst%": g["fwd12"].min() * 100,
        "nber_rec%": g["rec"].mean() * 100,
    })
    return t.reindex([n for n in names if n in t.index])


def window_ret(r, a, b):
    x = r.loc[a:b]
    return ((1 + x).prod() - 1) * 100 if len(x) else np.nan


# =====================================================
# MAIN
# =====================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="macro_dashboard.py")
    ap.add_argument("--start", default="1920-01-01")
    ap.add_argument("--min-cov", type=float, default=0.25,
                    help="min fraction of the 11 inputs present. 1920s have ~3, 1930s-40s ~5-6, 1950s+ ~7")
    ap.add_argument("--shiller", default=None, help="local path to Shiller ie_data.xls (else downloaded)")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sofr", default="DFF")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--legacy-alloc", action="store_true",
                    help="use the original 100/100/80/50/25/0 map instead of the dashboard's EXPOSURE")
    ap.add_argument("--out", default="backtest_long_out")
    a = ap.parse_args(argv)

    bm.HIST_START = "1920-01-01"
    bm.CACHE = Path(".bt_cache_long")
    mod = bm.load_dashboard(a.src)
    bm.apply_exposure(mod, a.legacy_alloc)
    Path(a.out).mkdir(exist_ok=True)

    print("Loading data...")
    gspc = bm.get_yahoo("^GSPC", a.refresh)
    if gspc.empty:
        sys.exit("^GSPC download failed")
    sh = get_shiller(a.shiller, a.refresh)
    px, ret = build_equity(gspc, sh)
    cal = px.index

    Y = {k: bm.get_yahoo(s, a.refresh) for k, s in mod.YAHOO_SYMBOLS.items()}
    F = {k: bm.get_fred(a.sofr if k == "sofr" else sid, a.refresh) for k, sid in mod.FRED_SERIES.items()}
    for sid in FRED_LONG:
        F[sid] = bm.get_fred(sid, a.refresh)

    rf_short = bm.pit(F["TB3MS"], cal, LAG["TB3MS"])
    gs10_sh = bm.pit(sh["GS10"].dropna(), cal, 35) if sh is not None else pd.Series(np.nan, index=cal)
    rf = (rf_short.fillna(gs10_sh - 1.0).clip(lower=0).fillna(0) / 100 / 12)
    rec = bm.on_cal(F["USREC"], cal).fillna(0)

    print("Scoring history (monthly)...")
    m, src = build_long_metrics(mod, cal, gspc, Y, F, sh)
    sig = bm.run_dashboard(mod, m)

    tr = (1 + ret.fillna(0)).cumprod()
    df = sig.copy()
    df["ret"], df["rf"], df["rec"] = ret, rf, rec
    df["fwd3"] = tr.shift(-3) / tr - 1
    df["fwd12"] = tr.shift(-12) / tr - 1

    sample = (df.index >= pd.Timestamp(a.start)) & (df["coverage"] >= a.min_cov) & df["ret"].notna()
    d = df[sample].copy()
    if d.empty:
        sys.exit("no months pass the coverage filter - lower --min-cov")
    print(f"\nSample: {d.index[0]:%Y-%m} -> {d.index[-1]:%Y-%m} ({len(d)} months), coverage >= {a.min_cov:.0%}")
    cov = (sig.loc[sig.index >= pd.Timestamp(a.start), "coverage"] * len(mod.RULES)).round()
    print("Avg inputs present (of %d) by decade:" % len(mod.RULES))
    print(cov.groupby(lambda x: f"{x.year // 10 * 10}s").mean().round(1).to_string())
    print("\nInput source share (% of months, proxy vs real) for proxied inputs:")
    print((src.loc[d.index].apply(lambda c: c.value_counts(normalize=True)).fillna(0).T * 100).round(0).to_string())

    names = [n for _, n in mod.LADDER] + ["RECESSION"]

    print("\n=== Forward total returns by CURRENT regime ===")
    print(regime_table(d, "cur_reg", names).round(2).to_string())
    pd.set_option("display.width", 200)
    for label, col in [("CURRENT regime", "cur_reg")]:
        print(f"\n=== Forward total returns by {label} (overlapping windows) ===")
        print(regime_table(d, col, names).round(2).to_string())
    print(f"\nUnconditional: fwd3m {d['fwd3'].mean() * 100:.2f}%  fwd12m {d['fwd12'].mean() * 100:.2f}%")

    print(f"\n=== Strategies (monthly rebalance, 1-month execution lag, {a.cost_bps}bps) ===")
    curves, wts = {"Buy&Hold": d["ret"]}, {}
    for label, col in [("Current", "cur_reg")]:
        w = d[col].map(bm.ALLOC).astype(float)
        r, pos, turn = bm.strat_returns(w, d["ret"], d["rf"], 1, a.cost_bps)
        curves[label], wts[label] = r, (pos, turn)
    common = pd.concat(curves, axis=1).dropna()

    def run(sub):
        out = {}
        for k in sub.columns:
            pos, turn = wts.get(k, (None, None))
            out[k] = stats(sub[k], d["rf"],
                           pos.reindex(sub.index) if pos is not None else None,
                           turn.reindex(sub.index) if turn is not None else None)
        return pd.DataFrame(out).T.round(2)

    print("\nFull sample:")
    print(run(common).to_string())
    for name, s, e in ERAS:
        sub = common.loc[s:e]
        if len(sub) > 24:
            print(f"\n{name} ({len(sub)} months):")
            print(run(sub).to_string())

    print("\n=== Drawdown episodes: total return % over window ===")
    rows = {n: {k: window_ret(common[k], s, e) for k in common.columns} for n, s, e in CRISES
            if len(common.loc[s:e])}
    print(pd.DataFrame(rows).T.round(1).to_string())

    d.to_csv(Path(a.out) / "monthly_signals.csv")
    src.to_csv(Path(a.out) / "input_sources.csv")
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
        ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Macro regime backtest, monthly (growth of $1)")
        ax[1].set_title("Drawdown")
        fig.tight_layout()
        fig.savefig(Path(a.out) / "equity_curves.png", dpi=130)
        print(f"\nSaved to {a.out}/")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")


if __name__ == "__main__":
    main()