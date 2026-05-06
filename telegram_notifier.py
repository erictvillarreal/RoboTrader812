"""
RoboTrader S4 — telegram_notifier.py
Mensajes estructurados: S4 TRADE | S4 DAILY | S4 RISK | S4 WEEKLY
Deduplicación por hash de contenido (evita duplicados por reinicios).
"""
import hashlib
import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, LOG_DIR

_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
_DEDUP_PATH = Path(LOG_DIR) / "notif_dedup.json"

# ── Deduplicación ─────────────────────────────────────────

def _load_dedup() -> set:
    if _DEDUP_PATH.exists():
        try:
            return set(json.loads(_DEDUP_PATH.read_text()))
        except Exception:
            pass
    return set()

def _save_dedup(seen: set):
    _DEDUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    # mantener solo las últimas 500
    lst = list(seen)[-500:]
    _DEDUP_PATH.write_text(json.dumps(lst))

def _dedup_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

# ── Core send ─────────────────────────────────────────────

def _send(text: str, deduplicate: bool = True) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[telegram] No config — mensaje omitido:\n{text[:80]}")
        return False

    key = _dedup_key(text)
    if deduplicate:
        seen = _load_dedup()
        if key in seen:
            print(f"[telegram] Dedup skip — key={key}")
            return False

    try:
        r = requests.post(
            _API + "/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        if deduplicate:
            seen = _load_dedup()
            seen.add(key)
            _save_dedup(seen)
        return True
    except Exception as e:
        print(f"[telegram] Error enviando: {e}")
        return False

# ── Mensaje helpers ───────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def send_trade(symbol: str, direction: str, entry: float,
               tp: float, sl: float, ev: float, p_up: float,
               stake_usdt: float, equity: float, mode: str = "paper"):
    """S4 TRADE — se envía en cada apertura de posición."""
    text = (
        f"<b>S4 TRADE</b> [{mode.upper()}]\n"
        f"Symbol:    <code>{symbol}</code>\n"
        f"Direction: <code>{direction.upper()}</code>\n"
        f"Entry:     <code>${entry:,.2f}</code>\n"
        f"TP:        <code>${tp:,.2f}</code>\n"
        f"SL:        <code>${sl:,.2f}</code>\n"
        f"EV:        <code>${ev:.4f}</code>\n"
        f"p_up:      <code>{p_up:.3f}</code>\n"
        f"Stake:     <code>${stake_usdt:.2f}</code>\n"
        f"Equity:    <code>${equity:,.2f}</code>\n"
        f"<i>{_now_utc()}</i>"
    )
    return _send(text, deduplicate=True)

def send_daily(equity: float, peak: float, trades_today: int,
               daily_pnl: float, budget_used_pct: float, mode: str = "paper"):
    """S4 DAILY — heartbeat al cambio de día UTC."""
    mdd_from_peak = (equity / peak - 1) * 100 if peak > 0 else 0
    text = (
        f"<b>S4 DAILY</b> [{mode.upper()}]\n"
        f"Equity:      <code>${equity:,.2f}</code>\n"
        f"Peak:        <code>${peak:,.2f}</code>\n"
        f"MDD vs peak: <code>{mdd_from_peak:.2f}%</code>\n"
        f"PnL hoy:     <code>${daily_pnl:+,.2f}</code>\n"
        f"Trades hoy:  <code>{trades_today}</code>\n"
        f"Budget used: <code>{budget_used_pct:.1f}%</code>\n"
        f"<i>{_now_utc()}</i>"
    )
    return _send(text, deduplicate=False)   # daily nunca dedup

def send_risk(event: str, equity: float, peak: float,
              detail: str = "", mode: str = "paper"):
    """S4 RISK — presupuesto agotado o kill-switch."""
    text = (
        f"<b>S4 RISK</b> [{mode.upper()}]\n"
        f"Event:  <code>{event}</code>\n"
        f"Equity: <code>${equity:,.2f}</code>\n"
        f"Peak:   <code>${peak:,.2f}</code>\n"
        f"MDD:    <code>{(equity/peak-1)*100:.2f}%</code>\n"
        + (f"Detail: {detail}\n" if detail else "")
        + f"<i>{_now_utc()}</i>"
    )
    return _send(text, deduplicate=False)

def send_weekly(equity: float, equity_start_week: float, trades_week: int,
                fees_week: float, win_rate: float, mode: str = "paper"):
    """S4 WEEKLY — resumen semanal."""
    pnl = equity - equity_start_week
    ret = pnl / equity_start_week * 100 if equity_start_week > 0 else 0
    text = (
        f"<b>S4 WEEKLY</b> [{mode.upper()}]\n"
        f"Equity:      <code>${equity:,.2f}</code>\n"
        f"PnL semana:  <code>${pnl:+,.2f} ({ret:+.2f}%)</code>\n"
        f"Trades:      <code>{trades_week}</code>\n"
        f"Fees:        <code>${fees_week:.2f}</code>\n"
        f"Win rate:    <code>{win_rate*100:.1f}%</code>\n"
        f"<i>{_now_utc()}</i>"
    )
    return _send(text, deduplicate=False)

def send_startup(equity: float, mode: str = "paper"):
    """Notificación de arranque del bot."""
    text = (
        f"<b>S4 START</b> [{mode.upper()}]\n"
        f"Equity inicial: <code>${equity:,.2f}</code>\n"
        f"<i>{_now_utc()}</i>"
    )
    return _send(text, deduplicate=False)

if __name__ == "__main__":
    print("[telegram] Módulo cargado. Para enviar, configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.")
    # Test sin config real — solo imprime el mensaje
    send_trade("BTCUSDT", "long", 65000, 66040, 64480,
               ev=12.34, p_up=0.62, stake_usdt=65.0, equity=1000.0, mode="paper")
