"""
RoboTrader S4 — trader.py
Loop principal 24/7. Corre como systemd service (robo-s4).

Modo paper:  MODE=paper  (default)
Modo live:   MODE=live   (requiere API keys y USE_TESTNET=false)

Flujo por vela:
  1. Descarga la vela más reciente cerrada
  2. Computa features
  3. Aplica s4_policy.decide()
  4. Si take=True: abre posición vía futures_broker
  5. Actualiza state.json + ledger
  6. Envía notificación Telegram
  7. Sleep hasta el cierre de la próxima vela
"""
import os
import sys
import time
import signal
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from config import (
    SYMBOL, INTERVAL, RAW_CSV, FEATURES,
    COMMISSION, SLIPPAGE, TP_MULT, SL_MULT, LEVERAGE,
)
from data_fetcher import get_historical_data
from tech_signals import add_technical_signals
from s4_policy import decide
from state import load as load_state, save as save_state, roll_day
from futures_broker import (
    initialize_symbol, open_long, open_short,
    get_balance, get_mark_price,
)
from telegram_notifier import (
    send_startup, send_trade, send_daily, send_risk,
)
from trade_logger import log_trade

# ── Config ────────────────────────────────────────────────

MODE           = os.getenv("MODE", "paper").lower()   # "paper" | "live"
PAPER          = (MODE == "paper")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")
LOOKBACK       = 300   # velas para features + ATR history
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "1000.0"))
HEARTBEAT_MIN  = int(os.getenv("HEARTBEAT_MIN", "60"))   # Telegram heartbeat cada N min

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("robo-s4")

# ── Shutdown handler ──────────────────────────────────────

_running = True

def _handle_signal(signum, frame):
    global _running
    log.info(f"Signal {signum} recibido — cerrando limpiamente...")
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ── Interval helpers ──────────────────────────────────────

def _interval_sec(interval: str) -> int:
    n = int(interval[:-1])
    return n * {"m": 60, "h": 3600, "d": 86400}[interval[-1]]

def _seconds_to_next_close(interval: str) -> float:
    now_ts  = datetime.now(timezone.utc).timestamp()
    step    = _interval_sec(interval)
    elapsed = now_ts % step
    return step - elapsed + 2   # +2s de margen para que la vela esté cerrada

# ── Data helpers ──────────────────────────────────────────

def _fetch_latest(symbol: str, interval: str, n: int = LOOKBACK) -> pd.DataFrame:
    df = get_historical_data(symbol, interval, limit=n)
    if df.empty or len(df) < 50:
        raise RuntimeError(f"Datos insuficientes: {len(df)} filas")
    df = add_technical_signals(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

# ── Daily rollover ────────────────────────────────────────

def _maybe_roll_day(state: dict, prev_day: str) -> tuple:
    today = datetime.now(timezone.utc).date().isoformat()
    if today != prev_day:
        equity     = state["equity"]
        peak       = state["peak_equity"]
        trades     = state["trades_today"]
        daily_pnl  = equity - state.get("day_open_equity", equity)
        budget_cap = state.get("day_open_equity", equity) * 0.005
        budget_used = max(0, state.get("day_open_equity", equity) - equity)
        budget_pct  = (budget_used / budget_cap * 100) if budget_cap > 0 else 0.0

        send_daily(equity, peak, trades, daily_pnl, budget_pct, mode=MODE)
        state = roll_day(state)
        save_state(state)
        log.info(f"Día nuevo: {today} | equity={equity:.2f} | pnl={daily_pnl:+.2f}")
    return state, today

# ── Main loop ─────────────────────────────────────────────

def main():
    global _running
    log.info(f"=== RoboTrader S4 arrancando | MODE={MODE.upper()} SYMBOL={SYMBOL} ===")

    # Inicializar
    initialize_symbol(SYMBOL, paper=PAPER)
    # Respetar INITIAL_EQUITY si el state.json no existe aún
    state    = roll_day(load_state())
    if state["equity"] == 1000.0 and INITIAL_EQUITY != 1000.0:
        state["equity"] = INITIAL_EQUITY
        state["peak_equity"] = INITIAL_EQUITY
        state["day_open_equity"] = INITIAL_EQUITY
    save_state(state)
    send_startup(state["equity"], mode=MODE)

    current_day = datetime.now(timezone.utc).date().isoformat()
    atr_history = []

    while _running:
        try:
            # ── Roll day check ────────────────────────────
            state, current_day = _maybe_roll_day(state, current_day)

            # ── Fetch + features ──────────────────────────
            df = _fetch_latest(SYMBOL, INTERVAL, LOOKBACK)

            # Usamos la vela -2 (penúltima = ya cerrada)
            row = df.iloc[-2].to_dict()
            atr_history = df["atr"].tolist()[-200:]

            # ── Kill-switch check ─────────────────────────
            mdd = state["equity"] / state["peak_equity"] - 1
            if mdd <= -0.25 and not state.get("kill_switch"):
                state["kill_switch"] = True
                save_state(state)
                send_risk("KILL_SWITCH", state["equity"], state["peak_equity"],
                          f"MDD {mdd*100:.2f}%", mode=MODE)
                log.warning(f"KILL SWITCH activado — MDD={mdd*100:.2f}%")

            # ── Policy decision ───────────────────────────
            # DEBUG TEMPORAL
            import numpy as np
            from config import FEATURES
            log.info(f"DEBUG row keys: {list(row.keys())[:8]}")
            log.info(f"DEBUG close={row.get('close','MISSING')} atr={row.get('atr','MISSING')}")
            feat_vals = {f: row.get(f, 'MISSING') for f in FEATURES}
            log.info(f"DEBUG features: {feat_vals}")
            d = decide(row, state, atr_history)
            log.info(
                f"Vela {row.get('open_time','?')} | "
                f"p_up={d.p_up:.3f} | take={d.take} dir={d.direction} "
                f"ev={d.ev:.4f} reason={d.reason}"
            )

            if d.take:
                close = float(row["close"])
                atr   = float(row["atr"])
                fees  = d.stake * LEVERAGE * (COMMISSION + SLIPPAGE) * 2

                # ── Ejecutar ──────────────────────────────
                if d.direction == "long":
                    result = open_long(
                        SYMBOL, d.stake, d.tp_price, d.sl_price, paper=PAPER
                    )
                else:
                    result = open_short(
                        SYMBOL, d.stake, d.tp_price, d.sl_price, paper=PAPER
                    )

                # En paper: simular PnL esperado (EV neto)
                pnl_sim = d.ev - fees if PAPER else 0.0

                new_equity = state["equity"] + (pnl_sim if PAPER else 0.0)
                new_equity = max(new_equity, 0.01)

                # Actualizar estado
                state["equity"]       = new_equity
                state["peak_equity"]  = max(state["peak_equity"], new_equity)
                state["trades_today"] = state.get("trades_today", 0) + 1
                state.setdefault("daily_evs", []).append(d.ev)
                save_state(state)

                # Ledger
                qty = d.stake * LEVERAGE / close
                log_trade(
                    SYMBOL, d.direction, close,
                    d.tp_price, d.sl_price, d.stake, qty,
                    d.p_up, d.ev, 1.0,
                    state["equity"] - pnl_sim, new_equity,
                    pnl_sim, fees,
                    outcome="paper" if PAPER else "open",
                    mode=MODE,
                )

                # Telegram
                send_trade(
                    SYMBOL, d.direction, close,
                    d.tp_price, d.sl_price,
                    d.ev, d.p_up, d.stake, new_equity, mode=MODE,
                )

                log.info(
                    f"TRADE {d.direction.upper()} | stake={d.stake:.2f} "
                    f"tp={d.tp_price:.2f} sl={d.sl_price:.2f} "
                    f"equity={new_equity:.2f}"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Error en loop: {e}", exc_info=True)

        # ── Sleep hasta siguiente cierre de vela ──────────
        sleep_s = _seconds_to_next_close(INTERVAL)
        log.info(f"Próxima vela en {sleep_s:.0f}s ({sleep_s/60:.1f} min)")

        # Sleep en chunks para responder a SIGTERM rápido
        deadline = time.time() + sleep_s
        while _running and time.time() < deadline:
            time.sleep(min(10, deadline - time.time()))

    log.info("=== RoboTrader S4 detenido limpiamente ===")

if __name__ == "__main__":
    main()