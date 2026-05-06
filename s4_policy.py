"""
RoboTrader S4 — s4_policy.py
Motor de decisión EV-first. Replica exactamente la lógica de walk.py
para uso en producción (una vela a la vez, sin look-ahead).

Entrada: features de la vela actual + estado actual
Salida:  Decision(take=bool, direction=str, stake=float, tp=float, sl=float, ev=float)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
import xgboost as xgb
from pathlib import Path

from config import (
    FEATURES, MODEL_PATH,
    MIN_P_LONG, LEVERAGE,
    POSITION_FRAC, POSITION_FRAC_MAX,
    MAX_TRADES_PER_DAY,
    COMMISSION, SLIPPAGE,
    TP_MULT, SL_MULT,
    VOL_SCALE_CLIP, VOL_PCTL, VOL_CUT_FACTOR,
    EV_MIN_PERC_STAKE, PROB_EDGE_MIN, EV_GAP_PERC,
    DAILY_EV_QUANTILE, MIN_OBS_FOR_Q,
    EV_CUSHION_MULT,
    RISK_DAILY_PCT, MDD_KILL_PCT, DAILY_BUDGET_IS_NET,
)

# ── Model singleton ───────────────────────────────────────

_model = None

def _load_model():
    global _model
    if _model is None:
        ubj = Path(MODEL_PATH).with_suffix(".ubj")
        pkl = Path(MODEL_PATH)
        if ubj.exists():
            clf = xgb.XGBClassifier()
            clf.load_model(str(ubj))
            _model = clf
        elif pkl.exists():
            import joblib
            _model = joblib.load(pkl)
        else:
            raise FileNotFoundError(f"Modelo no encontrado: {ubj} ni {pkl}")
    return _model

# ── Decision dataclass ────────────────────────────────────

@dataclass
class Decision:
    take:      bool
    direction: str        # "long" | "short" | "none"
    stake:     float      # USDT a arriesgar (sin leverage)
    tp_price:  float
    sl_price:  float
    ev:        float
    p_up:      float
    reason:    str        # por qué take=False (útil para logs)

_NO_TRADE = Decision(False, "none", 0.0, 0.0, 0.0, 0.0, 0.0, "")

# ── EV helpers ────────────────────────────────────────────

def _cost(price: float) -> float:
    return (COMMISSION + SLIPPAGE) * price

def _ev_long(p: float, tp_ret: float, sl_ret: float, entry: float) -> float:
    gross = p * tp_ret * entry - (1 - p) * abs(sl_ret) * entry
    cost  = (COMMISSION + SLIPPAGE) * entry * 2
    return gross - cost

def _ev_short(p: float, tp_ret: float, sl_ret: float, entry: float) -> float:
    p_down = 1 - p
    gross  = p_down * tp_ret * entry - p * abs(sl_ret) * entry
    cost   = (COMMISSION + SLIPPAGE) * entry * 2
    return gross - cost

# ── Volatility sizing ─────────────────────────────────────

def _vol_scale(atr: float, close: float, atr_history: list) -> float:
    if len(atr_history) < 10:
        return 1.0
    pctl   = np.percentile(atr_history, VOL_PCTL * 100)
    ratio  = atr / (close + 1e-12)
    pctl_r = pctl / (close + 1e-12)
    scale  = pctl_r / (ratio + 1e-12)
    if ratio > pctl_r * VOL_CUT_FACTOR:
        scale *= 0.5
    return float(np.clip(scale, *VOL_SCALE_CLIP))

# ── Risk checks ───────────────────────────────────────────

def _check_kill(state: dict) -> bool:
    equity = state["equity"]
    peak   = state["peak_equity"]
    return (equity / peak - 1) <= -MDD_KILL_PCT

def _check_daily_budget(state: dict) -> bool:
    """True si aún hay presupuesto disponible."""
    equity     = state["equity"]
    day_open   = state.get("day_open_equity", equity)
    budget     = day_open * RISK_DAILY_PCT
    if DAILY_BUDGET_IS_NET:
        used = max(0.0, day_open - equity)
    else:
        daily_evs = state.get("daily_evs", [])
        used = sum(e for e in daily_evs if e < 0)
    return used < budget

# ── Daily EV quantile filter (causal) ────────────────────

def _ev_quantile_ok(ev: float, daily_evs: list) -> bool:
    if len(daily_evs) < MIN_OBS_FOR_Q:
        return True
    threshold = np.quantile(daily_evs, DAILY_EV_QUANTILE)
    return ev >= threshold

# ── Main decision function ────────────────────────────────

def decide(row: dict, state: dict, atr_history: list) -> Decision:
    """
    row: dict con al menos FEATURES + 'close' + 'atr'
    state: dict de state.py
    atr_history: lista de ATR recientes (últimas ~200 velas)
    """
    # ── Guard 0: Kill-switch global ───────────────────────
    if state.get("kill_switch", False) or _check_kill(state):
        return Decision(False, "none", 0, 0, 0, 0, 0, "kill_switch")

    # ── Guard 1: Max trades por día ───────────────────────
    if state.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
        return Decision(False, "none", 0, 0, 0, 0, 0, "max_trades_today")

    # ── Guard 2: Presupuesto diario ───────────────────────
    if not _check_daily_budget(state):
        return Decision(False, "none", 0, 0, 0, 0, 0, "daily_budget_exhausted")

    # ── Features → probabilidad ───────────────────────────
    try:
        X = np.array([[row[f] for f in FEATURES]], dtype=np.float32)
    except KeyError as e:
        return Decision(False, "none", 0, 0, 0, 0, 0, f"missing_feature:{e}")

    model = _load_model()
    p_up  = float(model.predict_proba(X)[0, 1])

    close = float(row["close"])
    atr   = float(row["atr"])

    tp_ret = TP_MULT * atr / close
    sl_ret = SL_MULT * atr / close

    # ── Dirección ─────────────────────────────────────────
    if p_up >= MIN_P_LONG:
        direction = "long"
        ev = _ev_long(p_up, tp_ret, sl_ret, close)
    else:
        direction = "short"
        p_down = 1 - p_up
        ev = _ev_short(p_up, tp_ret, sl_ret, close)

    # ── Filtros EV ────────────────────────────────────────
    equity = state["equity"]

    # PROB_EDGE: diferencia mínima sobre 0.5
    prob_edge = abs(p_up - 0.5)
    if prob_edge < PROB_EDGE_MIN:
        return Decision(False, "none", 0, 0, 0, ev, p_up, "prob_edge_low")

    # EV_MIN_PERC_STAKE
    ev_perc = ev / (equity * POSITION_FRAC + 1e-12)
    if ev_perc < EV_MIN_PERC_STAKE * EV_CUSHION_MULT:
        return Decision(False, "none", 0, 0, 0, ev, p_up, "ev_min_perc_low")

    # EV_GAP (EV vs costo puro)
    cost_only = (COMMISSION + SLIPPAGE) * close * 2
    if ev < cost_only * (1 + EV_GAP_PERC):
        return Decision(False, "none", 0, 0, 0, ev, p_up, "ev_gap_low")

    # Daily EV quantile (causal)
    daily_evs = state.get("daily_evs", [])
    if not _ev_quantile_ok(ev, daily_evs):
        return Decision(False, "none", 0, 0, 0, ev, p_up, "daily_ev_quantile")

    # ── Sizing por volatilidad ────────────────────────────
    scale    = _vol_scale(atr, close, atr_history)
    base_frac = POSITION_FRAC * scale
    frac     = min(base_frac, POSITION_FRAC_MAX)
    stake    = equity * frac          # USDT sin leverage

    # ── Precios TP/SL ─────────────────────────────────────
    if direction == "long":
        tp_price = close + TP_MULT * atr
        sl_price = close - SL_MULT * atr
    else:
        tp_price = close - TP_MULT * atr
        sl_price = close + SL_MULT * atr

    return Decision(
        take=True,
        direction=direction,
        stake=stake,
        tp_price=tp_price,
        sl_price=sl_price,
        ev=ev,
        p_up=p_up,
        reason="ok",
    )

if __name__ == "__main__":
    import json
    from state import load, roll_day

    state = roll_day(load())

    # Fila sintética para smoke test
    test_row = {
        "ema_10": 65000, "ema_30": 64800, "rsi_14": 58.0,
        "macd": 120.0, "macd_signal": 90.0, "macd_diff": 30.0,
        "atr": 800.0, "close": 65000.0,
    }
    atr_hist = [700.0 + i * 2 for i in range(100)]

    d = decide(test_row, state, atr_hist)
    print(f"Decision: take={d.take} dir={d.direction} stake={d.stake:.2f} "
          f"ev={d.ev:.4f} p_up={d.p_up:.3f} reason={d.reason}")
