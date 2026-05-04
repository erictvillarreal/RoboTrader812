"""
RoboTrader S4 — trade_logger.py
Escribe cada trade al ledger CSV. Reproducible y auditable.
"""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from config import LEDGER_PATH

_COLS = [
    "ts", "symbol", "direction", "entry_price",
    "tp_price", "sl_price", "stake_usdt", "qty",
    "p_up", "ev", "vol_scale",
    "equity_before", "equity_after", "pnl", "fees",
    "outcome", "mode", "reason",
]

def _ensure_header(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(_COLS)

def log_trade(
    symbol: str, direction: str, entry_price: float,
    tp_price: float, sl_price: float, stake_usdt: float, qty: float,
    p_up: float, ev: float, vol_scale: float,
    equity_before: float, equity_after: float, pnl: float, fees: float,
    outcome: str,   # "tp" | "sl" | "open" | "paper"
    mode: str = "paper",
    reason: str = "",
):
    path = Path(LEDGER_PATH)
    _ensure_header(path)
    ts = datetime.now(timezone.utc).isoformat()
    row = [
        ts, symbol, direction, entry_price,
        tp_price, sl_price, stake_usdt, qty,
        round(p_up, 5), round(ev, 6), round(vol_scale, 4),
        round(equity_before, 4), round(equity_after, 4),
        round(pnl, 4), round(fees, 4),
        outcome, mode, reason,
    ]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

if __name__ == "__main__":
    log_trade(
        "BTCUSDT", "long", 65000.0, 66600.0, 64480.0,
        65.0, 0.002, 0.62, 12.5, 1.05,
        1000.0, 1065.0, 65.0, 1.3,
        "tp", "paper",
    )
    print(f"Log escrito en {LEDGER_PATH}")
