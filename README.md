# RoboTrader S4

Sistema algorítmico EV-first para BTCUSDT Futures. Walk-forward causal, kill-switch, Telegram notifications.

## Deploy en Railway (30 días paper)

### 1. Variables de entorno requeridas

```
MODE=paper
USE_TESTNET=false
BOT_SYMBOL=BTCUSDT
BOT_TIMEFRAME=1h
INITIAL_EQUITY=1000.0
HEARTBEAT_MIN=60
LOG_LEVEL=INFO

# Binance (no se usan en paper mode pero deben existir)
BINANCE_API_KEY=placeholder
BINANCE_API_SECRET=placeholder

# Telegram (REQUERIDO para ver PnL)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 2. Crear bot de Telegram

1. Hablar con @BotFather en Telegram → `/newbot`
2. Guardar el token
3. Enviar un mensaje al bot, luego ir a:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Copiar el `chat_id` del resultado

### 3. Deploy en Railway

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Crear proyecto
railway init

# Deploy
railway up

# Ver logs en vivo
railway logs
```

### 4. Qué verás en Telegram

- **S4 START** — cuando el bot arranca
- **S4 TRADE** — cada vez que abre posición (dirección, TP, SL, EV, equity)
- **S4 DAILY** — cada cambio de día UTC (equity, PnL del día, trades)
- **S4 RISK** — si se activa el kill-switch o se agota el budget diario

### 5. Métricas a revisar cada semana

| Métrica | Target | Alerta |
|---------|--------|--------|
| Win rate | >60% | <50% por 50+ trades |
| Expectancy/trade | >$1.50 | <$0.50 |
| Tracking error | <30% | >30% → considerar retrain |
| MDD | <5% | >10% → revisar |

### 6. Estructura del proyecto

```
trader.py          # Loop principal 24/7
config.py          # Parámetros S4_causal_safe
s4_policy.py       # Motor de decisión EV-first
futures_broker.py  # Ejecución Binance Futures
telegram_notifier.py
state.py           # Estado persistente
trade_logger.py    # Ledger CSV
live_validator.py  # Validación estadística
model/
  best_model.ubj   # XGBoost entrenado (walk-forward)
```

## Parámetros S4_causal_safe

| Parámetro | Valor |
|-----------|-------|
| LEVERAGE | 2x |
| POSITION_FRAC | 6.5% |
| MAX_TRADES_PER_DAY | 2 |
| RISK_DAILY_PCT | 0.5% |
| MDD_KILL_PCT | 25% |
| TP_MULT | 2.0 × ATR |
| SL_MULT | 0.8 × ATR |
| COMMISSION | 10 bps |
| SLIPPAGE | 2 bps |

## Backtest (S4_causal_safe, Oct 2022 – Sep 2025)

- CAGR: **78.2%**
- MDD: **−0.93%**
- Sharpe: **19.5** (inflado por MTM; real estimado 3–8)
- Win rate: **69%**
- Trades: **2,084** (~2/día)
- Kill-switch: **nunca activado**
