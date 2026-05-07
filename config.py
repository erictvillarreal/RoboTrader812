"""
RoboTrader S4 — config.py
Single source of truth. All modules import CONFIG from here.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Exchange ──────────────────────────────────────────────
FUTURES_BASE   = os.getenv("BINANCE_FUTS_BASE", "https://fapi.binance.com")
TESTNET_BASE   = os.getenv("BINANCE_TESTNET_BASE", "https://testnet.binancefuture.com")
USE_TESTNET    = os.getenv("USE_TESTNET", "true").lower() in ("1", "true", "yes")

API_KEY        = os.getenv("BINANCE_API_KEY", "")
API_SECRET     = os.getenv("BINANCE_API_SECRET", "")

# ── Symbol & Timeframe ────────────────────────────────────
SYMBOL         = os.getenv("BOT_SYMBOL", "BTCUSDT")
INTERVAL       = os.getenv("BOT_TIMEFRAME", "1h")

# ── Paths ─────────────────────────────────────────────────
DATA_DIR       = BASE_DIR / "data"
MODEL_DIR      = BASE_DIR / "model"
LOG_DIR        = BASE_DIR / "logs"
VAR_DIR        = BASE_DIR / "var"

RAW_CSV        = DATA_DIR / f"{SYMBOL}.csv"
LABELED_CSV    = DATA_DIR / f"{SYMBOL}_labeled.csv"
MODEL_PATH     = MODEL_DIR / "best_model.pkl"
STATE_PATH     = VAR_DIR  / "state.json"
LEDGER_PATH    = LOG_DIR  / "trade_ledger.csv"

# ── Triple Barrera ────────────────────────────────────────
TP_MULT        = 2.0
SL_MULT        = 0.8
HORIZON        = 12     # velas

# ── Walk-Forward ──────────────────────────────────────────
WINDOW_DAYS    = 180
STEP_DAYS      = 14

# ── Features (orden fijo — no cambiar sin re-etiquetar) ───
FEATURES = ["ema_10", "ema_30", "rsi_14", "macd", "macd_signal", "macd_diff", "atr"]

# ── Costos realistas ──────────────────────────────────────
COMMISSION     = 0.0010   # 10 bps taker futures
SLIPPAGE       = 0.0002   # 2 bps estimado

# ── Política / Sizing ─────────────────────────────────────
LEVERAGE           = 2.0
POSITION_FRAC      = 0.065
POSITION_FRAC_MAX  = 0.13
MAX_TRADES_PER_DAY = 2
MIN_P_LONG         = 0.55
EV_CUSHION_MULT    = 1.0

# Sizing por volatilidad
VOL_SCALE_CLIP  = (0.6, 1.5)
VOL_PCTL        = 0.95
VOL_CUT_FACTOR  = 0.75

# Filtros EV
EV_MIN_PERC_STAKE  = 0.003
PROB_EDGE_MIN      = 0.04
EV_GAP_PERC        = 0.0000  # desactivado — EV_MIN_PERC_STAKE ya cubre este filtro
DAILY_EV_QUANTILE  = 0.20
MIN_OBS_FOR_Q      = 6

# ── Marco de riesgo ───────────────────────────────────────
RISK_DAILY_PCT      = 0.005   # S4_causal_safe params
MDD_KILL_PCT        = 0.25
DAILY_BUDGET_IS_NET = True

# ── Walk model ────────────────────────────────────────────
N_ESTIMATORS   = 1200
LEARNING_RATE  = 0.03
VAL_SPLIT      = 0.15

# ── Telegram ──────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Alias dict para compatibilidad con scripts legacy ─────
CONFIG = {
    "futures_base":        FUTURES_BASE,
    "testnet_base":        TESTNET_BASE,
    "use_testnet":         USE_TESTNET,
    "api_key":             API_KEY,
    "api_secret":          API_SECRET,
    "symbol":              SYMBOL,
    "interval":            INTERVAL,
    "raw_csv":             str(RAW_CSV),
    "labeled_csv":         str(LABELED_CSV),
    "model_path":          str(MODEL_PATH),
    "state_path":          str(STATE_PATH),
    "ledger_path":         str(LEDGER_PATH),
    "tp_mult":             TP_MULT,
    "sl_mult":             SL_MULT,
    "horizon":             HORIZON,
    "window_days":         WINDOW_DAYS,
    "step_days":           STEP_DAYS,
    "features":            FEATURES,
    "commission":          COMMISSION,
    "slippage":            SLIPPAGE,
    "leverage":            LEVERAGE,
    "position_frac":       POSITION_FRAC,
    "position_frac_max":   POSITION_FRAC_MAX,
    "max_trades_per_day":  MAX_TRADES_PER_DAY,
    "min_p_long":          MIN_P_LONG,
    "risk_daily_pct":      RISK_DAILY_PCT,
    "mdd_kill_pct":        MDD_KILL_PCT,
    "telegram_token":      TELEGRAM_TOKEN,
    "telegram_chat_id":    TELEGRAM_CHAT_ID,
}
