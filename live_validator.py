"""
RoboTrader S4 — live_validator.py
Validacion estadistica paper vs live con intervalos de confianza al 95%.
Detecta regime shifts, tracking error, y degradacion de senal.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime, timezone

PAPER_BENCHMARKS = {
    "sharpe":         19.51,
    "cagr":           78.17,
    "win_rate":        0.674,
    "expectancy":      1.977,
    "daily_vol":       0.0401,
    "mdd":             1.021,
    "trades_per_day":  2.09,
}

class LiveValidator:
    def __init__(self, paper_report_path, live_ledger_path, min_trades=30):
        self.paper      = json.loads(Path(paper_report_path).read_text())
        raw             = pd.read_csv(live_ledger_path)
        self.min_trades = min_trades
        self.df         = self._prep(raw)

    def _prep(self, raw):
        df      = raw.copy()
        ts_col  = "ts" if "ts" in df.columns else "open_time"
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df["date"] = df["ts"].dt.date
        pnl_col = "pnl_simulated" if "pnl_simulated" in df.columns else "pnl"
        df["pnl"] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
        win_col = "outcome" if "outcome" in df.columns else "win"
        w = df[win_col]
        if w.dtype == object:
            df["win"] = w.map({"tp": True, "sl": False, "timeout": False,
                               "paper": True, "True": True, "False": False,
                               True: True, False: False}).fillna(False)
        else:
            df["win"] = w.astype(bool)
        return df.sort_values("ts").reset_index(drop=True)

    def _daily_returns(self):
        daily = self.df.groupby("date")["pnl"].sum()
        eq    = 1000.0 + daily.cumsum()
        return eq.pct_change().dropna()

    def win_rate_test(self):
        wins  = int(self.df["win"].sum())
        n     = len(self.df)
        p     = wins / n if n > 0 else 0.0
        z     = 1.96
        denom  = 1 + z**2 / n
        center = (p + z**2/(2*n)) / denom
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        p_ref  = PAPER_BENCHMARKS["win_rate"]
        se     = np.sqrt(p_ref*(1-p_ref)/n)
        z_stat = (p - p_ref) / se if se > 0 else 0.0
        p_val  = 2*(1 - stats.norm.cdf(abs(z_stat)))
        return {"win_rate": round(p,4), "ci_95": (round(center-margin,4), round(center+margin,4)),
                "paper_ref": p_ref, "z_stat": round(z_stat,3),
                "p_value": round(p_val,4), "sig": p_val < 0.05, "n": n}

    def expectancy_test(self):
        pnl   = self.df["pnl"].values
        n     = len(pnl)
        mean  = pnl.mean()
        se    = pnl.std(ddof=1)/np.sqrt(n) if n > 1 else 0.0
        p_ref = PAPER_BENCHMARKS["expectancy"]
        t_stat, p_val = stats.ttest_1samp(pnl, p_ref)
        return {"expectancy": round(mean,4), "ci_95": (round(mean-1.96*se,4), round(mean+1.96*se,4)),
                "paper_ref": p_ref, "t_stat": round(t_stat,3),
                "p_value": round(p_val,4), "sig": p_val < 0.05, "n": n}

    def sharpe_test(self):
        dr = self._daily_returns()
        n  = len(dr)
        if n < 5:
            return {"error": f"solo {n} dias"}
        mu, sd = dr.mean(), dr.std(ddof=1)
        sr = mu/sd*np.sqrt(365) if sd > 0 else 0.0
        se = np.sqrt((1 + 0.5*sr**2)/n)
        return {"sharpe_live": round(sr,2), "ci_95": (round(sr-1.96*se,2), round(sr+1.96*se,2)),
                "paper_ref": PAPER_BENCHMARKS["sharpe"], "n_days": n,
                "note": "MTM correcto - vol real declarada"}

    def tracking_error(self):
        n       = len(self.df)
        ev_real = self.df["pnl"].sum()
        n_paper = self.paper.get("total_trades", 2236)
        ev_exp  = self.paper.get("net_pnl", 4419.52) * (n/n_paper)
        if abs(ev_exp) < 1e-9:
            return {"error": "EV esperado cero"}
        te  = abs(ev_real - ev_exp)/abs(ev_exp)*100
        pnl_arr = self.df["pnl"].values
        boots = [abs(np.random.choice(pnl_arr,n,replace=True).sum()-ev_exp)/abs(ev_exp)*100
                 for _ in range(1000)]
        ci = (round(np.percentile(boots,2.5),2), round(np.percentile(boots,97.5),2))
        return {"tracking_error_pct": round(te,2), "ci_95": ci,
                "ev_expected": round(ev_exp,2), "ev_realized": round(ev_real,2),
                "direction": "under" if ev_real < ev_exp else "over",
                "alert_30": te > 30, "retrain_urgent_50": te > 50, "n": n}

    def regime_ks_test(self):
        pnl  = self.df["pnl"].values
        n    = len(pnl)
        if n < 20:
            return {"error": f"insuficientes trades ({n})"}
        half = n//2
        early, late = pnl[:half], pnl[half:]
        ks_stat, p_val = stats.ks_2samp(early, late)
        drift = late.mean() - early.mean()
        return {"ks_stat": round(ks_stat,4), "p_value": round(p_val,4),
                "shift": p_val < 0.05, "mean_early": round(early.mean(),4),
                "mean_late": round(late.mean(),4), "drift": round(drift,4),
                "direction": "degrading" if drift < 0 else "improving", "n": n}

    def drawdown_live(self):
        eq   = 1000.0 + self.df["pnl"].cumsum()
        peak = eq.cummax()
        dd   = (eq/peak) - 1
        mdd  = dd.min()
        return {"mdd_live": round(mdd*100,4), "paper_ref": -PAPER_BENCHMARKS["mdd"],
                "in_dd_pct": round((dd < -0.001).sum()/len(dd)*100,1),
                "alert": abs(mdd*100) > 5.0, "kill_zone": abs(mdd*100) > 20.0}

    def full_report(self, verbose=True):
        n = len(self.df)
        if n < self.min_trades:
            msg = f"Solo {n} trades - minimo {self.min_trades} para significancia"
            if verbose: print(f"  {msg}")
            return {"warning": msg, "n": n}
        result = {
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "n_trades":       n,
            "win_rate":       self.win_rate_test(),
            "expectancy":     self.expectancy_test(),
            "sharpe":         self.sharpe_test(),
            "tracking_error": self.tracking_error(),
            "regime":         self.regime_ks_test(),
            "drawdown":       self.drawdown_live(),
        }
        if verbose:
            self._print(result)
        return result

    def _print(self, r):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{'='*56}")
        print(f"  S4 Live Validation - {ts}")
        print(f"  Trades: {r['n_trades']}")
        print(f"{'='*56}")
        wr = r["win_rate"]
        flag = "DIVERGE p<0.05" if wr["sig"] else "OK"
        print(f"\n-- Win Rate")
        print(f"  Live:  {wr['win_rate']*100:.2f}%  CI95:[{wr['ci_95'][0]*100:.1f}%, {wr['ci_95'][1]*100:.1f}%]")
        print(f"  Paper: {wr['paper_ref']*100:.2f}%   Status: {flag}")
        ex = r["expectancy"]
        flag = "DIVERGE p<0.05" if ex["sig"] else "OK"
        print(f"\n-- Expectancy/trade")
        print(f"  Live:  ${ex['expectancy']:.4f}  CI95:[${ex['ci_95'][0]:.3f}, ${ex['ci_95'][1]:.3f}]")
        print(f"  Paper: ${ex['paper_ref']:.4f}   Status: {flag}")
        sh = r["sharpe"]
        if "error" not in sh:
            print(f"\n-- Sharpe (MTM correcto)")
            print(f"  Live:  {sh['sharpe_live']:.2f}  CI95:[{sh['ci_95'][0]:.2f}, {sh['ci_95'][1]:.2f}]")
            print(f"  Paper: {sh['paper_ref']:.2f} (inflado - vol subdeclarada en backtest)")
        te = r["tracking_error"]
        if "error" not in te:
            flag = "RETRAIN URGENTE" if te["retrain_urgent_50"] else ("ALERTA" if te["alert_30"] else "OK")
            print(f"\n-- Tracking Error: {te['tracking_error_pct']:.2f}%  [{te['ci_95'][0]:.1f}%, {te['ci_95'][1]:.1f}%]  {flag}")
            print(f"  EV esperado: ${te['ev_expected']:.2f}  EV real: ${te['ev_realized']:.2f}  ({te['direction']})")
        reg = r["regime"]
        if "error" not in reg:
            flag = "SHIFT DETECTADO" if reg["shift"] else "estable"
            print(f"\n-- Regimen (KS): stat={reg['ks_stat']:.4f} p={reg['p_value']:.4f}  {flag}")
            print(f"  Drift: ${reg['drift']:+.4f}/trade ({reg['direction']})")
        dd = r["drawdown"]
        flag = "KILL ZONE" if dd["kill_zone"] else ("ALERTA" if dd["alert"] else "OK")
        print(f"\n-- Drawdown: MDD={dd['mdd_live']:.4f}%  {flag}  (paper: {dd['paper_ref']:.4f}%)")
        print()


if __name__ == "__main__":
    from config import LOG_DIR
    v = LiveValidator(str(Path(LOG_DIR)/"paper_report.json"),
                      str(Path(LOG_DIR)/"paper_ledger.csv"), min_trades=30)
    v.full_report()
