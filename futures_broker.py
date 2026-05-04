"""
RoboTrader S4 — futures_broker.py
Capa de ejecución para Binance Futures USDT-M.
Soporta: paper mode, testnet, live.
Coloca MARKET entry + SL/TP como órdenes separadas (STOP_MARKET + TAKE_PROFIT_MARKET).
"""
import os
import math
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone
from typing import Optional

from config import (
    API_KEY, API_SECRET, USE_TESTNET,
    FUTURES_BASE, TESTNET_BASE,
    LEVERAGE, SYMBOL,
)

_BASE = TESTNET_BASE if USE_TESTNET else FUTURES_BASE

# ── Auth helpers ──────────────────────────────────────────

def _sign(params: dict) -> dict:
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = hmac.new(
        API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    params["signature"] = sig
    return params

def _headers() -> dict:
    return {"X-MBX-APIKEY": API_KEY}

def _ts() -> int:
    return int(time.time() * 1000)

def _get(path: str, params: dict = None, signed: bool = False):
    params = params or {}
    if signed:
        params["timestamp"] = _ts()
        params = _sign(params)
    r = requests.get(_BASE + path, params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def _post(path: str, params: dict):
    params["timestamp"] = _ts()
    params = _sign(params)
    r = requests.post(_BASE + path, params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def _delete(path: str, params: dict):
    params["timestamp"] = _ts()
    params = _sign(params)
    r = requests.delete(_BASE + path, params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

# ── Account info ──────────────────────────────────────────

def get_balance(asset: str = "USDT") -> float:
    """Retorna el balance disponible (walletBalance) en USDT."""
    data = _get("/fapi/v2/account", signed=True)
    for a in data.get("assets", []):
        if a["asset"] == asset:
            return float(a["walletBalance"])
    return 0.0

def get_mark_price(symbol: str = SYMBOL) -> float:
    data = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return float(data["markPrice"])

def get_position(symbol: str = SYMBOL) -> dict:
    """Retorna la posición actual (puede ser flat)."""
    data = _get("/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
    for p in data:
        if p["symbol"] == symbol:
            return {
                "size":        float(p["positionAmt"]),
                "entry_price": float(p["entryPrice"]),
                "unrealized":  float(p["unRealizedProfit"]),
                "leverage":    float(p["leverage"]),
            }
    return {"size": 0.0, "entry_price": 0.0, "unrealized": 0.0, "leverage": 1.0}

# ── Precision helpers ─────────────────────────────────────

_exchange_info_cache = {}

def _get_symbol_info(symbol: str) -> dict:
    global _exchange_info_cache
    if symbol not in _exchange_info_cache:
        data = _get("/fapi/v1/exchangeInfo")
        for s in data["symbols"]:
            if s["symbol"] == symbol:
                _exchange_info_cache[symbol] = s
                break
    return _exchange_info_cache.get(symbol, {})

# BTCUSDT Futures defaults (step=0.001 BTC, tick=$0.10)
_QTY_DEFAULTS  = {"BTCUSDT": (0.001, 3)}
_PRICE_DEFAULTS = {"BTCUSDT": (0.10, 1)}

def _round_qty(symbol: str, qty: float) -> float:
    try:
        info = _get_symbol_info(symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                precision = max(0, round(-math.log10(step)))
                return round(math.floor(qty / step) * step, precision)
    except Exception:
        pass
    step, prec = _QTY_DEFAULTS.get(symbol, (0.001, 3))
    return round(math.floor(qty / step) * step, prec)

def _round_price(symbol: str, price: float) -> float:
    try:
        info = _get_symbol_info(symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
                precision = max(0, round(-math.log10(tick)))
                return round(round(price / tick) * tick, precision)
    except Exception:
        pass
    tick, prec = _PRICE_DEFAULTS.get(symbol, (0.10, 1))
    return round(round(price / tick) * tick, prec)

# ── Leverage setup ────────────────────────────────────────

def set_leverage(symbol: str = SYMBOL, leverage: int = int(LEVERAGE)):
    try:
        return _post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})
    except Exception as e:
        print(f"[broker] set_leverage error (puede ya estar seteado): {e}")

def set_margin_type(symbol: str = SYMBOL, margin_type: str = "ISOLATED"):
    try:
        return _post("/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})
    except Exception as e:
        # -4046 = ya está en ese modo — ignorar
        if "-4046" not in str(e):
            print(f"[broker] set_margin_type: {e}")

# ── Order placement ───────────────────────────────────────

def _market_order(symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
    params = {
        "symbol":   symbol,
        "side":     side,        # BUY | SELL
        "type":     "MARKET",
        "quantity": qty,
    }
    if reduce_only:
        params["reduceOnly"] = "true"
    return _post("/fapi/v1/order", params)

def _stop_order(symbol: str, side: str, qty: float,
                stop_price: float, order_type: str = "STOP_MARKET") -> dict:
    """
    order_type: STOP_MARKET (SL) | TAKE_PROFIT_MARKET (TP)
    """
    params = {
        "symbol":     symbol,
        "side":       side,
        "type":       order_type,
        "quantity":   qty,
        "stopPrice":  stop_price,
        "reduceOnly": "true",
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }
    return _post("/fapi/v1/order", params)

def cancel_all_orders(symbol: str = SYMBOL):
    try:
        return _delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
    except Exception as e:
        print(f"[broker] cancel_all_orders: {e}")

# ── High-level trade entry ────────────────────────────────

def open_long(symbol: str, stake_usdt: float, tp_price: float,
              sl_price: float, paper: bool = True, mock_price: float = 0.0) -> dict:
    """
    Abre LONG con MARKET + coloca TP y SL como reduce-only.
    stake_usdt: capital sin apalancamiento (broker aplica leverage).
    """
    mark = mock_price if (paper and mock_price > 0) else get_mark_price(symbol)
    notional = stake_usdt * LEVERAGE
    raw_qty  = notional / mark
    qty      = _round_qty(symbol, raw_qty)
    tp_r     = _round_price(symbol, tp_price)
    sl_r     = _round_price(symbol, sl_price)

    if paper:
        result = {
            "mode":       "paper",
            "symbol":     symbol,
            "side":       "BUY",
            "qty":        qty,
            "mark_price": mark,
            "tp":         tp_r,
            "sl":         sl_r,
            "stake_usdt": stake_usdt,
            "notional":   notional,
            "ts":         datetime.now(timezone.utc).isoformat(),
        }
        print(f"[PAPER LONG] {symbol} qty={qty} @~{mark:.2f} | TP={tp_r} SL={sl_r}")
        return result

    # Live
    entry = _market_order(symbol, "BUY", qty)
    tp_ord = _stop_order(symbol, "SELL", qty, tp_r, "TAKE_PROFIT_MARKET")
    sl_ord = _stop_order(symbol, "SELL", qty, sl_r, "STOP_MARKET")
    print(f"[LIVE LONG] {symbol} qty={qty} @~{mark:.2f} | TP={tp_r} SL={sl_r}")
    return {"entry": entry, "tp_order": tp_ord, "sl_order": sl_ord}

def open_short(symbol: str, stake_usdt: float, tp_price: float,
               sl_price: float, paper: bool = True, mock_price: float = 0.0) -> dict:
    """
    Abre SHORT con MARKET + coloca TP y SL como reduce-only.
    """
    mark = mock_price if (paper and mock_price > 0) else get_mark_price(symbol)
    notional = stake_usdt * LEVERAGE
    raw_qty  = notional / mark
    qty      = _round_qty(symbol, raw_qty)
    tp_r     = _round_price(symbol, tp_price)
    sl_r     = _round_price(symbol, sl_price)

    if paper:
        result = {
            "mode":       "paper",
            "symbol":     symbol,
            "side":       "SELL",
            "qty":        qty,
            "mark_price": mark,
            "tp":         tp_r,
            "sl":         sl_r,
            "stake_usdt": stake_usdt,
            "notional":   notional,
            "ts":         datetime.now(timezone.utc).isoformat(),
        }
        print(f"[PAPER SHORT] {symbol} qty={qty} @~{mark:.2f} | TP={tp_r} SL={sl_r}")
        return result

    # Live
    entry = _market_order(symbol, "SELL", qty)
    tp_ord = _stop_order(symbol, "BUY", qty, tp_r, "TAKE_PROFIT_MARKET")
    sl_ord = _stop_order(symbol, "BUY", qty, sl_r, "STOP_MARKET")
    print(f"[LIVE SHORT] {symbol} qty={qty} @~{mark:.2f} | TP={tp_r} SL={sl_r}")
    return {"entry": entry, "tp_order": tp_ord, "sl_order": sl_ord}

def close_position(symbol: str = SYMBOL, paper: bool = True) -> dict:
    """Cierra la posición abierta y cancela órdenes pendientes."""
    if paper:
        print(f"[PAPER CLOSE] {symbol}")
        return {"mode": "paper", "symbol": symbol, "action": "close"}
    cancel_all_orders(symbol)
    pos = get_position(symbol)
    size = pos["size"]
    if abs(size) < 1e-6:
        return {"msg": "no position"}
    side = "SELL" if size > 0 else "BUY"
    qty  = _round_qty(symbol, abs(size))
    return _market_order(symbol, side, qty, reduce_only=True)

# ── Setup helper (llamar una vez al iniciar el bot) ───────

def initialize_symbol(symbol: str = SYMBOL, paper: bool = True):
    if paper:
        print(f"[broker] Paper mode — skip leverage/margin setup")
        return
    set_margin_type(symbol, "ISOLATED")
    set_leverage(symbol, int(LEVERAGE))
    print(f"[broker] {symbol} — ISOLATED, {int(LEVERAGE)}x leverage set")

if __name__ == "__main__":
    print("futures_broker.py — smoke test (paper)")
    print(f"  BASE: {_BASE}")
    print(f"  USE_TESTNET: {USE_TESTNET}")
    print(f"  LEVERAGE: {LEVERAGE}")
