"""
RoboTrader S4 — metrics_reporter.py
Genera reporte diario: equity, tracking error, EV realizado vs esperado,
drawdown, y alerta si el sistema se está desviando del backtest.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import LOG_DIR, VAR_DIR

# Benchmarks del backtest S4_causal_safe
BACKTEST_BENCHMARKS = {
    "win_rate":         0.690,
    "expectancy":       2.36,
    "daily_ret_mean":   0.00208,   # ~78% CAGR / 365
    "daily_vol":        0.00110,
    "max_tracking_err": 30.0,      # % — alerta si supera
}

def load_paper_report() -> dict:
    p = Path(LOG_DIR) / "paper_report.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

def load_paper_ledger() -> pd.DataFrame:
    p = Path(LOG_DIR) / "paper_ledger.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def generate_daily_report() -> dict:
    report  = load_paper_report()
    ledger  = load_paper_ledger()

    if not report or ledger.empty:
        return {"status": "sin datos — corre paper_engine.py primero"}

    ledger["ts"] = pd.to_datetime(ledger["ts"], utc=True, errors="coerce")
    ledger["date"] = ledger["ts"].dt.date

    # ── Métricas rolling 30 días ──────────────────────────
    cutoff_30 = ledger["ts"].max() - pd.Timedelta(days=30)
    last_30   = ledger[ledger["ts"] >= cutoff_30]

    win_30    = (last_30["outcome"] == "tp").mean() if len(last_30) > 0 else 0
    exp_30    = last_30["pnl_simulated"].mean() if len(last_30) > 0 else 0
    fees_30   = last_30["fees"].sum()
    trades_30 = len(last_30)

    # ── Equity curve y ratios ─────────────────────────────
    daily_eq = ledger.groupby("date")["equity_after"].last()
    if len(daily_eq) > 1:
        daily_ret = daily_eq.pct_change().dropna()
        ann_vol   = daily_ret.std() * np.sqrt(365) * 100
        sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(365)
                     if daily_ret.std() > 0 else 0)
    else:
        ann_vol, sharpe = 0.0, 0.0

    # ── Tracking error (últimos 30 días) ──────────────────
    tracking_err = report.get("tracking_error", 0.0)

    # ── Alertas ───────────────────────────────────────────
    alerts = []
    if tracking_err > BACKTEST_BENCHMARKS["max_tracking_err"]:
        alerts.append(f"TRACKING ERROR {tracking_err:.1f}% > 30% — considerar re-entrenamiento")
    if win_30 < BACKTEST_BENCHMARKS["win_rate"] - 0.15 and trades_30 > 20:
        alerts.append(f"WIN RATE 30d ({win_30*100:.1f}%) diverge del backtest ({BACKTEST_BENCHMARKS['win_rate']*100:.0f}%)")
    if report.get("kill_switch"):
        alerts.append("KILL SWITCH ACTIVADO — bot detenido")
    if report.get("mdd", 0) < -10:
        alerts.append(f"MDD {report['mdd']:.2f}% — revisar parámetros")

    result = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "status":            "ALERT" if alerts else "OK",
        "alerts":            alerts,
        "equity":            report.get("equity"),
        "net_pnl":           report.get("net_pnl"),
        "cagr":              report.get("cagr_annualized"),
        "mdd":               report.get("mdd"),
        "tracking_error":    tracking_err,
        "total_trades":      report.get("trades_total"),
        "win_rate_all":      report.get("win_rate"),
        "win_rate_30d":      round(win_30 * 100, 2),
        "trades_30d":        trades_30,
        "fees_30d":          round(fees_30, 2),
        "expectancy_30d":    round(exp_30, 4),
        "ann_vol":           round(ann_vol, 4),
        "sharpe":            round(sharpe, 2),
        "benchmarks":        BACKTEST_BENCHMARKS,
    }
    return result

def print_daily_report():
    r = generate_daily_report()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*46}")
    print(f"  S4 Daily Metrics Report — {now}")
    print(f"{'='*46}")
    print(f"  Status:          {'🔴 ALERT' if r.get('status')=='ALERT' else '🟢 OK'}")
    if r.get("alerts"):
        for a in r["alerts"]:
            print(f"  ⚠️  {a}")
    print()
    print(f"  Equity:          ${r.get('equity', 0):>10,.2f}")
    print(f"  Net PnL:         ${r.get('net_pnl', 0):>+10,.2f}")
    print(f"  CAGR:            {r.get('cagr', 0):>+9.2f}%")
    print(f"  MDD:             {r.get('mdd', 0):>+9.4f}%")
    print()
    print(f"  Tracking error:  {r.get('tracking_error', 0):>9.2f}%  (limit: 30%)")
    print(f"  Win rate (all):  {r.get('win_rate_all', 0):>9.2f}%  (backtest: 69.0%)")
    print(f"  Win rate (30d):  {r.get('win_rate_30d', 0):>9.2f}%")
    print(f"  Expectancy 30d:  ${r.get('expectancy_30d', 0):>9.4f}")
    print(f"  Trades total:    {r.get('total_trades', 0):>10}")
    print(f"  Trades 30d:      {r.get('trades_30d', 0):>10}")
    print(f"  Fees 30d:        ${r.get('fees_30d', 0):>10,.2f}")
    print(f"  Sharpe (live):   {r.get('sharpe', 0):>10.2f}")
    print(f"  Ann. vol:        {r.get('ann_vol', 0):>9.4f}%")
    print()
    return r

if __name__ == "__main__":
    print_daily_report()
