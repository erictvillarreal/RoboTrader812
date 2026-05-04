"""
RoboTrader S4 — paper_engine.py
Motor de paper trading sobre datos históricos reales.
Simula el loop vela-por-vela exactamente como lo haría trader.py en vivo,
pero usando el equity_intraday.csv + walk_report.csv ya validados.

Genera:
  - var/paper_state.json  (estado acumulado)
  - logs/paper_ledger.csv (trades simulados)
  - logs/paper_report.json (métricas de sesión)

Uso:
  python paper_engine.py              # corre todas las velas disponibles
  python paper_engine.py --days 30    # últimos 30 días
  python paper_engine.py --report     # solo imprime el reporte actual
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SYMBOL, FEATURES,
    TP_MULT, SL_MULT,
    COMMISSION, SLIPPAGE, LEVERAGE,
    POSITION_FRAC, POSITION_FRAC_MAX,
    MAX_TRADES_PER_DAY,
    MIN_P_LONG,
    RISK_DAILY_PCT, MDD_KILL_PCT,
    EV_MIN_PERC_STAKE, PROB_EDGE_MIN, EV_GAP_PERC,
    DAILY_EV_QUANTILE, MIN_OBS_FOR_Q,
    EV_CUSHION_MULT, VOL_SCALE_CLIP, VOL_PCTL, VOL_CUT_FACTOR,
    DATA_DIR, LOG_DIR, VAR_DIR,
)
from tech_signals import add_technical_signals
from s4_policy import decide, _load_model

PAPER_STATE_PATH  = Path(VAR_DIR)  / "paper_state.json"
PAPER_LEDGER_PATH = Path(LOG_DIR)  / "paper_ledger.csv"
PAPER_REPORT_PATH = Path(LOG_DIR)  / "paper_report.json"

LEDGER_COLS = [
    "ts","symbol","direction","entry_price","tp_price","sl_price",
    "stake_usdt","qty","p_up","ev","equity_before","equity_after",
    "pnl_simulated","fees","outcome",
]

# ── State ─────────────────────────────────────────────────

def _default_state(equity: float = 1000.0) -> dict:
    return {
        "equity": equity, "peak_equity": equity,
        "trades_total": 0, "wins": 0, "losses": 0,
        "trades_today": 0, "daily_evs": [],
        "day_open_equity": equity, "current_day": None,
        "kill_switch": False, "fees_total": 0.0,
        "ev_realized_total": 0.0, "ev_expected_total": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_bar": None,
    }

def _load_paper_state() -> dict:
    if PAPER_STATE_PATH.exists():
        return json.loads(PAPER_STATE_PATH.read_text())
    return _default_state()

def _save_paper_state(s: dict):
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_PATH.write_text(json.dumps(s, indent=2, default=str))

def _roll_day(s: dict, today: str) -> dict:
    if s.get("current_day") != today:
        s["trades_today"]    = 0
        s["daily_evs"]       = []
        s["day_open_equity"] = s["equity"]
        s["current_day"]     = today
    return s

# ── Ledger ────────────────────────────────────────────────

def _ensure_ledger():
    p = PAPER_LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        pd.DataFrame(columns=LEDGER_COLS).to_csv(p, index=False)

def _append_trade(row: dict):
    _ensure_ledger()
    pd.DataFrame([row]).to_csv(
        PAPER_LEDGER_PATH, mode="a", header=False, index=False
    )

# ── Simulation helpers ────────────────────────────────────

def _simulate_outcome(direction: str, entry: float, tp: float, sl: float,
                      future_highs: np.ndarray, future_lows: np.ndarray) -> str:
    """Determina si se tocó TP o SL primero en las próximas velas."""
    for h, l in zip(future_highs, future_lows):
        if direction == "long":
            if h >= tp: return "tp"
            if l <= sl: return "sl"
        else:
            if l <= tp: return "tp"
            if h >= sl: return "sl"
    return "timeout"

# ── Main engine ───────────────────────────────────────────

def run_paper(df: pd.DataFrame, initial_equity: float = 1000.0,
              reset: bool = False, verbose: bool = True) -> dict:
    """
    Corre el paper engine sobre el DataFrame de velas.
    df debe tener: open_time, open, high, low, close, volume + FEATURES + atr
    Retorna dict de métricas finales.
    """
    model = _load_model()
    state = _default_state(initial_equity) if reset else _load_paper_state()
    _ensure_ledger()

    df = df.sort_values("open_time").reset_index(drop=True)
    n  = len(df)

    # Determinar desde dónde continuar
    last_bar = state.get("last_bar")
    if last_bar and not reset:
        df = df[df["open_time"].astype(str) > last_bar].reset_index(drop=True)
        if verbose:
            print(f"[paper] Continuando desde {last_bar} — {len(df)} velas nuevas")

    if len(df) < 20:
        if verbose:
            print(f"[paper] Solo {len(df)} velas — sin datos suficientes")
        return _build_report(state)

    atr_history = df["atr"].tolist()[:200]

    trades_this_run = 0
    for i in range(20, len(df) - 12):   # -12 para poder simular outcome
        bar       = df.iloc[i]
        today_str = str(bar["open_time"])[:10]
        state     = _roll_day(state, today_str)

        # Kill-switch check
        mdd = state["equity"] / state["peak_equity"] - 1
        if mdd <= -MDD_KILL_PCT:
            state["kill_switch"] = True
            if verbose:
                print(f"[paper] KILL SWITCH — MDD={mdd*100:.2f}%")
            break

        row = {f: bar[f] for f in FEATURES}
        row["close"] = float(bar["close"])
        row["atr"]   = float(bar["atr"])
        atr_history  = df["atr"].iloc[max(0, i-200):i].tolist()

        d = decide(row, state, atr_history)

        if not d.take:
            continue

        # Simular outcome con velas futuras
        future = df.iloc[i+1:i+13]
        outcome = _simulate_outcome(
            d.direction,
            float(bar["close"]), d.tp_price, d.sl_price,
            future["high"].values, future["low"].values,
        )

        entry   = float(bar["close"])
        fees    = d.stake * LEVERAGE * (COMMISSION + SLIPPAGE) * 2

        if outcome == "tp":
            gross_pnl = d.stake * LEVERAGE * TP_MULT * float(bar["atr"]) / entry
            win = True
        elif outcome == "sl":
            gross_pnl = -d.stake * LEVERAGE * SL_MULT * float(bar["atr"]) / entry
            win = False
        else:  # timeout — neutral
            gross_pnl = 0.0
            win = False

        net_pnl     = gross_pnl - fees
        eq_before   = state["equity"]
        new_equity  = max(eq_before + net_pnl, 0.01)
        qty         = d.stake * LEVERAGE / entry

        # Update state
        state["equity"]              = new_equity
        state["peak_equity"]         = max(state["peak_equity"], new_equity)
        state["trades_today"]        = state.get("trades_today", 0) + 1
        state["trades_total"]        = state.get("trades_total", 0) + 1
        state["fees_total"]          = state.get("fees_total", 0.0) + fees
        state["ev_expected_total"]   = state.get("ev_expected_total", 0.0) + d.ev
        state["ev_realized_total"]   = state.get("ev_realized_total", 0.0) + net_pnl
        state.setdefault("daily_evs", []).append(d.ev)
        state["last_bar"] = str(bar["open_time"])

        if win:
            state["wins"] = state.get("wins", 0) + 1
        else:
            state["losses"] = state.get("losses", 0) + 1

        trades_this_run += 1

        _append_trade({
            "ts":            str(bar["open_time"]),
            "symbol":        SYMBOL,
            "direction":     d.direction,
            "entry_price":   round(entry, 2),
            "tp_price":      round(d.tp_price, 2),
            "sl_price":      round(d.sl_price, 2),
            "stake_usdt":    round(d.stake, 4),
            "qty":           round(qty, 6),
            "p_up":          round(d.p_up, 5),
            "ev":            round(d.ev, 6),
            "equity_before": round(eq_before, 4),
            "equity_after":  round(new_equity, 4),
            "pnl_simulated": round(net_pnl, 4),
            "fees":          round(fees, 4),
            "outcome":       outcome,
        })

        if verbose and trades_this_run % 10 == 0:
            print(f"  [{trades_this_run:4d} trades] eq=${new_equity:,.2f} "
                  f"win={state['wins']}/{state['trades_total']} "
                  f"mdd={mdd*100:.2f}%")

    _save_paper_state(state)
    report = _build_report(state)
    _save_report(report)
    return report

# ── Report ────────────────────────────────────────────────

def _build_report(state: dict) -> dict:
    total    = state.get("trades_total", 0)
    wins     = state.get("wins", 0)
    losses   = state.get("losses", 0)
    equity   = state.get("equity", 1000.0)
    peak     = state.get("peak_equity", 1000.0)
    fees     = state.get("fees_total", 0.0)
    ev_exp   = state.get("ev_expected_total", 0.0)
    ev_real  = state.get("ev_realized_total", 0.0)

    win_rate      = wins / total if total > 0 else 0.0
    tracking_err  = abs(ev_real - ev_exp) / abs(ev_exp) if abs(ev_exp) > 1e-9 else 0.0
    mdd           = equity / peak - 1 if peak > 0 else 0.0
    net_pnl       = equity - 1000.0
    expectancy    = net_pnl / total if total > 0 else 0.0

    # CAGR desde started_at
    started = state.get("started_at", datetime.now(timezone.utc).isoformat())
    try:
        t0    = datetime.fromisoformat(started)
        years = (datetime.now(timezone.utc) - t0).total_seconds() / (365.25 * 86400)
        cagr  = (equity / 1000.0) ** (1 / years) - 1 if years > 0.01 else 0.0
    except Exception:
        cagr = 0.0

    return {
        "equity":          round(equity, 2),
        "peak_equity":     round(peak, 2),
        "net_pnl":         round(net_pnl, 2),
        "cagr_annualized": round(cagr * 100, 2),
        "mdd":             round(mdd * 100, 4),
        "trades_total":    total,
        "wins":            wins,
        "losses":          losses,
        "win_rate":        round(win_rate * 100, 2),
        "fees_total":      round(fees, 2),
        "ev_expected":     round(ev_exp, 4),
        "ev_realized":     round(ev_real, 4),
        "tracking_error":  round(tracking_err * 100, 2),
        "expectancy_per_trade": round(expectancy, 4),
        "kill_switch":     state.get("kill_switch", False),
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }

def _save_report(report: dict):
    PAPER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_REPORT_PATH.write_text(json.dumps(report, indent=2))

def print_report(report: dict):
    print()
    print("╔══════════════════════════════════════╗")
    print("║     RoboTrader S4 — Paper Report     ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Equity:          ${report['equity']:>10,.2f}   (inicio $1,000)")
    print(f"  Net PnL:         ${report['net_pnl']:>+10,.2f}")
    print(f"  CAGR:            {report['cagr_annualized']:>+9.2f}%")
    print(f"  MDD:             {report['mdd']:>+9.4f}%")
    print()
    print(f"  Trades:          {report['trades_total']:>10}")
    print(f"  Win rate:        {report['win_rate']:>9.1f}%")
    print(f"  Expectancy/trade:${report['expectancy_per_trade']:>9.4f}")
    print(f"  Fees:            ${report['fees_total']:>10,.2f}")
    print()
    print(f"  EV expected:     {report['ev_expected']:>10.2f}")
    print(f"  EV realized:     {report['ev_realized']:>10.2f}")
    print(f"  Tracking error:  {report['tracking_error']:>9.2f}%")
    print(f"  Kill switch:     {str(report['kill_switch']):>10}")
    print(f"  Generated:       {report['generated_at'][:19]} UTC")
    print()

# ── CLI ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",   type=int, default=0, help="Últimos N días (0=todos)")
    ap.add_argument("--reset",  action="store_true", help="Reiniciar estado desde cero")
    ap.add_argument("--report", action="store_true", help="Solo mostrar reporte actual")
    ap.add_argument("--equity", type=float, default=1000.0, help="Equity inicial")
    ap.add_argument("--quiet",  action="store_true")
    args = ap.parse_args()

    if args.report:
        if PAPER_REPORT_PATH.exists():
            print_report(json.loads(PAPER_REPORT_PATH.read_text()))
        else:
            print("No hay reporte aún. Corre primero: python paper_engine.py")
        return

    # Cargar labeled dataset
    labeled_path = Path(DATA_DIR) / f"BTCUSDT_labeled.csv"
    if not labeled_path.exists():
        # Fallback: reconstruir desde equity_intraday si no hay labeled CSV
        print(f"[paper] {labeled_path} no encontrado.")
        print("[paper] Para un run completo necesitas el dataset etiquetado.")
        print("[paper] Corre: python update_data.py && python make_labeled_dataset.py")
        return

    df = pd.read_csv(labeled_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = add_technical_signals(df) if "ema_10" not in df.columns else df
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if args.days > 0:
        cutoff = df["open_time"].max() - pd.Timedelta(days=args.days)
        df = df[df["open_time"] >= cutoff].reset_index(drop=True)
        print(f"[paper] Filtrando últimos {args.days} días — {len(df)} velas")

    print(f"[paper] Iniciando sobre {len(df)} velas "
          f"({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})")

    report = run_paper(df, initial_equity=args.equity,
                       reset=args.reset, verbose=not args.quiet)
    print_report(report)

if __name__ == "__main__":
    main()
