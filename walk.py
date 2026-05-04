# walk.py    Walk-forward con dataset pre-etiquetado + equity diaria vs intrada
#            + Marco de riesgo (presupuesto diario / kill-switch)
#            + Cuantil CAUSAL online (T2 fix)
#            + Export detallado para auditora (T6/T7: train_* / embargo_sec / bar_seconds)
from pathlib import Path
from datetime import timedelta
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# =========================
# Config por defecto (ajustable)
# =========================
SYMBOL = "BTCUSDT"
DATA_PATH = Path(__file__).resolve().parent / "data" / f"{SYMBOL}_labeled.csv"

# Ventana / Paso
WINDOW_DAYS = 180
STEP_DAYS   = 14

# Triple Barrera (solo para magnitudes de TP/SL; el label ya viene en el CSV)
TP_MULT = 2.0
SL_MULT = 0.8
HORIZON = 12  # velas

# Costos realistas
COMMISSION = 0.0010
SLIPPAGE   = 0.0002

# Calidad de seal / Exposicin
MAX_TRADES_PER_DAY = 2
POSITION_FRAC      = 0.065
POSITION_FRAC_MAX  = 0.13
EV_MIN_RET         = 0.0

# Sizing por volatilidad
VOL_SCALE_CLIP = (0.6, 1.5)
VOL_PCTL       = 0.95
VOL_CUT_FACTOR = 0.75

# Validacin interna
VAL_SPLIT     = 0.15
N_ESTIMATORS  = 1200
LEARNING_RATE = 0.03

FEATURES = ["ema_10", "ema_30", "rsi_14", "macd", "macd_signal", "macd_diff", "atr"]

EV_CUSHION_MULT = 1.0
MIN_P_LONG = 0.55

# Filtros EV (ajusta en tus barridas)
EV_MIN_PERC_STAKE = 0.003
PROB_EDGE_MIN     = 0.04
EV_GAP_PERC       = 0.0005
DAILY_EV_QUANTILE = 0.20     # umbral de cuantil diario CAUSAL
MIN_OBS_FOR_Q     = 6        # warm-up causal: hasta acumular N obs, no se filtra por cuantil
LEVERAGE          = 2.0

# ===== Marco de riesgo =====
RISK_DAILY_PCT     = 0.005   # S4_causal_safe: presupuesto diario
MDD_KILL_PCT       = 0.25    # S4_causal_safe: kill-switch global
DAILY_BUDGET_IS_NET = True   # presupuesto diario vs equity de apertura del da (True) o dinmico (False)

# =========================
# Utilidades
# =========================
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_features(df: pd.DataFrame):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features: {missing}. Ajusta tu backtester para generarlas.")
    return df

def _build_model(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

def _fit_with_calibration(tr: pd.DataFrame):
    X = tr[FEATURES]
    y = tr["label"].astype(int)

    cut = int(len(X) * (1.0 - VAL_SPLIT))
    cut = max(100, min(cut, len(X) - 50))
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_va, y_va = X.iloc[cut:], y.iloc[cut:]

    early_stop = xgb.callback.EarlyStopping(rounds=100, save_best=True)
    model = _build_model()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, callbacks=[early_stop])

    calibrated = None
    try:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated.fit(X_va, y_va)
        proba_test = calibrated.predict_proba(X_va)[:, 1]
        if np.allclose(np.nanstd(proba_test), 0.0):
            calibrated = None
    except Exception:
        calibrated = None
    return calibrated, model

def _predict_proba(model, calibrated, X):
    return calibrated.predict_proba(X)[:, 1] if calibrated is not None else model.predict_proba(X)[:, 1]

def _vol_metrics(train_slice: pd.DataFrame):
    atr_ratio = (train_slice["atr"] / train_slice["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ref = float(atr_ratio.median()) if len(atr_ratio) > 0 else 0.0
    extreme = float(atr_ratio.quantile(VOL_PCTL)) if len(atr_ratio) > 10 else np.inf
    return ref, extreme

def _vol_position_scale(atr_over_close: float, ref: float, clip_range=VOL_SCALE_CLIP) -> float:
    tiny = 1e-9
    if ref <= tiny or atr_over_close <= tiny:
        return 1.0
    raw = np.sqrt(ref / max(atr_over_close, tiny))
    return float(np.clip(raw, clip_range[0], clip_range[1]))

def _apply_costs(stake_lev: float, commission=COMMISSION, slippage=SLIPPAGE) -> float:
    return stake_lev * (commission + slippage)

def _cap_trades_by_day(dates: pd.Series, max_per_day=MAX_TRADES_PER_DAY):
    counts, mask = {}, []
    for ts in dates:
        d = pd.to_datetime(ts).date()
        n = counts.get(d, 0)
        if n < max_per_day:
            mask.append(True); counts[d] = n + 1
        else:
            mask.append(False)
    return np.array(mask, dtype=bool)

def _estimate_bar_seconds(df_time: pd.Series) -> int:
    s = pd.to_datetime(df_time).sort_values()
    dif = s.diff().dropna().dt.total_seconds()
    return int(np.median(dif)) if len(dif) > 0 else 3600

def _embargo_timedelta(train_df: pd.DataFrame, H=HORIZON) -> tuple[timedelta, int]:
    ts = pd.to_datetime(train_df["open_time"]).sort_values()
    if len(ts) >= 2:
        bar_seconds = _estimate_bar_seconds(ts)
    else:
        bar_seconds = 3600
    return timedelta(seconds=int(H * bar_seconds)), bar_seconds

def _purge_by_embargo(test_df: pd.DataFrame, train_end_ts: pd.Timestamp, embargo: timedelta) -> pd.DataFrame:
    thr = pd.to_datetime(train_end_ts) + embargo
    return test_df[test_df["open_time"] > thr].copy()

def _ensure_returns_columns(df: pd.DataFrame):
    # crea ret_tp/sl_* slo con ATR y close si faltan (sin mirar futuro)
    if not {"ret_tp_long", "ret_sl_long"}.issubset(df.columns):
        df["ret_tp_long"] = (df["close"] + TP_MULT * df["atr"]) / df["close"] - 1.0
        df["ret_sl_long"] = (df["close"] - SL_MULT * df["atr"]) / df["close"] - 1.0  # negativo
    if not {"ret_tp_short", "ret_sl_short"}.issubset(df.columns):
        df["ret_tp_short"] = 1.0 - (df["close"] - TP_MULT * df["atr"]) / df["close"]
        df["ret_sl_short"] = -((df["close"] + SL_MULT * df["atr"]) / df["close"] - 1.0)
    for c in ["ret_tp_long", "ret_sl_long", "ret_tp_short", "ret_sl_short"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

# =========================
# Core: Walk-Forward con EV + reporting diario/intradiario + RISK FRAMEWORK + T2/T6/T7 auditora
# =========================
def run_walk(
    symbol=SYMBOL,
    window_days=WINDOW_DAYS,
    step_days=STEP_DAYS,
    position_frac=POSITION_FRAC,
    position_frac_max=POSITION_FRAC_MAX,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    max_trades_per_day=MAX_TRADES_PER_DAY,
    ev_min_ret=EV_MIN_RET,
    ev_min_perc_stake=EV_MIN_PERC_STAKE,
    prob_edge_min=PROB_EDGE_MIN,
    ev_gap_perc=EV_GAP_PERC,
    daily_ev_quantile=DAILY_EV_QUANTILE,
    min_obs_for_q=MIN_OBS_FOR_Q,
    leverage=LEVERAGE,
    # ====== RIESGO ======
    risk_daily_pct=RISK_DAILY_PCT,
    mdd_kill_pct=MDD_KILL_PCT,
    daily_budget_is_net=DAILY_BUDGET_IS_NET,
    data_path=DATA_PATH,
):
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_model.pkl"

    # === Carga dataset congelado ===
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontr el dataset pre-etiquetado: {data_path}")
    df = pd.read_csv(data_path)

    # Normaliza tiempo a naive/UTC para evitar tz-mix
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    n_rows = len(df)
    sha = _sha256_file(data_path)
    print(f"[INFO] Dataset: {data_path} | filas={n_rows} | sha256={sha[:12]}&")

    # Tipos y saneo
    for col in ["close", "atr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "label" not in df.columns:
        raise ValueError("El dataset debe incluir columna 'label' (0/1) generada SIN look-ahead por el backtester.")
    _ensure_features(df)
    _ensure_returns_columns(df)

    df = df.sort_values("open_time").reset_index(drop=True)

    start_date = df["open_time"].min()
    end_date   = df["open_time"].max()
    t0   = start_date + timedelta(days=window_days)
    step = timedelta(days=step_days)

    # Tracking
    equity0 = 1000.0
    equity  = equity0
    equity_curve = []   # intrada (por trade)
    rows = []
    total_fees = 0.0

    # === Estado de riesgo ===
    peak_equity = equity
    kill_switch_triggered = False
    risk_caps_applied = 0
    risk_skips = 0
    days_paused = 0

    # Diario / B&H (para comparativas, NO para riesgo)
    daily_close = df[["open_time", "close"]].copy()
    daily_close["date"] = daily_close["open_time"].dt.floor("D")
    px_daily = daily_close.groupby("date")["close"].last().dropna()

    # Helpers de riesgo
    def _update_peak(eq: float):
        nonlocal peak_equity
        if eq > peak_equity:
            peak_equity = eq

    def _global_floor() -> float:
        return peak_equity * (1.0 - float(mdd_kill_pct))

    # Estado de presupuesto diario
    current_day = None
    day_open_equity = equity
    day_floor_equity = day_open_equity * (1.0 - float(risk_daily_pct))

    def _maybe_roll_day(ts: pd.Timestamp):
        """Resetea presupuesto diario al cambiar de da calendario."""
        nonlocal current_day, day_open_equity, day_floor_equity, days_paused
        d = pd.to_datetime(ts).date()
        if current_day is None or d != current_day:
            current_day = d
            day_open_equity = equity
            day_floor_equity = day_open_equity * (1.0 - float(risk_daily_pct))
            return True
        return False

    def _remaining_daily_loss() -> float:
        if daily_budget_is_net:
            return max(0.0, equity - day_floor_equity)
        else:
            dyn_floor = equity * (1.0 - float(risk_daily_pct))
            return max(0.0, equity - dyn_floor)

    def _stake_cap_from_budget(ret_sl_eff: float, fee_pct: float) -> float:
        """Cap de stake (SIN leverage) por presupuesto diario restante y kill global."""
        tiny = 1e-12
        r_worst_per_unlev = float(leverage) * (abs(ret_sl_eff) + fee_pct)
        if r_worst_per_unlev <= tiny:
            return np.inf
        rem_day = _remaining_daily_loss()
        rem_glob = max(0.0, equity - _global_floor())
        cap_day = rem_day / r_worst_per_unlev
        cap_glob = rem_glob / r_worst_per_unlev
        return max(0.0, min(cap_day, cap_glob))

    # ===== Export acumulado para auditora de candidatos (T2/T6/T7) =====
    cand_exports = []  # por-slice, luego concatenamos y guardamos

    t = t0
    while t <= end_date:
        train_start = t - timedelta(days=window_days)
        train_end   = t
        test_end    = min(t + step, end_date)

        tr = df[(df["open_time"] > train_start) & (df["open_time"] <= train_end)].copy()
        te = df[(df["open_time"] > train_end)  & (df["open_time"] <= test_end)].copy()
        if len(tr) < 200 or len(te) == 0:
            t += step
            continue

        # Embargo por H
        embargo_td, bar_seconds = _embargo_timedelta(tr, HORIZON)
        te = _purge_by_embargo(te, train_end, embargo_td)
        purged_n = int(((df["open_time"] > train_end) & (df["open_time"] <= test_end)).sum() - len(te))

        X_tr, y_tr = tr[FEATURES], tr["label"].astype(int)
        X_te, y_te = te[FEATURES], te["label"].astype(int)

        # Entrena y calibra
        calibrated, model = _fit_with_calibration(tr)
        try:
            joblib.dump(calibrated if calibrated is not None else model, model_path)
        except Exception:
            pass

        # Probas calibradas
        p_up = _predict_proba(model, calibrated, X_te).astype(float)
        p_up   = np.clip(p_up, 1e-6, 1 - 1e-6)
        p_down = 1.0 - p_up

        # Mtricas informativas
        preds_up = (p_up >= 0.5).astype(int)
        acc = accuracy_score(y_te, preds_up)
        p, r, f1, _ = precision_recall_fscore_support(y_te, preds_up, average="binary", zero_division=0)

        # Volatilidad para sizing base
        ref_vol, extreme_vol = _vol_metrics(tr)

        # ====== Seleccin por EV (vectorizada) ======
        te = te.copy()
        te["p_up"]   = p_up
        te["p_down"] = p_down

        atr_ratio = (te["atr"] / te["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scale_vol = atr_ratio.apply(lambda x: _vol_position_scale(float(x), ref_vol))

        # stake0 (sin leverage) para umbrales relativos
        stake0 = equity * float(position_frac) * scale_vol
        stake0 = np.where(atr_ratio >= float(extreme_vol), stake0 * float(VOL_CUT_FACTOR), stake0)
        stake0 = np.minimum(stake0, equity * float(position_frac_max))
        stake0 = np.maximum(stake0, 0.0)

        # stake con leverage para EV/fees (solo para clculo de EV)
        stakeL   = stake0 * float(leverage)
        fee_pct  = float(commission + slippage)
        fees_vec = stakeL * fee_pct

        # EV long / short (dinero, con leverage)
        te["ev_long"] = stakeL * (te["p_up"]   * te["ret_tp_long"]   + (1.0 - te["p_up"])   * te["ret_sl_long"])  - fees_vec
        te["ev_shrt"] = stakeL * (te["p_down"] * te["ret_tp_short"]  + (1.0 - te["p_down"]) * te["ret_sl_short"]) - fees_vec

        te["direction"] = np.where(te["ev_long"] >= te["ev_shrt"], "long", "short")
        te["ev_best"]   = np.where(te["ev_long"] >= te["ev_shrt"], te["ev_long"], te["ev_shrt"])
        te["ev_gap"]    = np.abs(te["ev_long"] - te["ev_shrt"])
        prob_edge       = np.abs(te["p_up"] - 0.5)

        # Filtros base (edge/EV/GAP)
        min_ev_stake      = stake0 * float(ev_min_perc_stake)
        min_ev_eq         = equity * float(ev_min_ret) if ev_min_ret is not None else 0.0
        min_ev_money_vec  = np.maximum(min_ev_stake, min_ev_eq)
        mask_edge         = (te["ev_best"] >= min_ev_money_vec) & (prob_edge >= float(prob_edge_min))
        mask_gap          = mask_edge & (te["ev_gap"] >= (stake0 * float(ev_gap_perc)))

        # === Cuantil CAUSAL (online) por da ===
        cand = te[mask_gap].copy()
        cand["day"] = cand["open_time"].dt.date
        cand = cand.sort_values(["day", "open_time"])  # orden temporal

        # running-threshold por da
        q_keep_list = []
        q_thr_list  = []
        if cand.empty or daily_ev_quantile <= 0.0:
            cand["q_keep"] = True
            cand["q_thr"]  = np.nan
        else:
            q = float(daily_ev_quantile)
            m = int(min_obs_for_q)
            for d, g in cand.groupby("day", sort=False):
                evs = g["ev_best"].to_numpy()
                keep = []
                qthr = []
                for i in range(len(g)):
                    if i+1 < m:
                        keep.append(True); qthr.append(np.nan)  # warm-up: no filtramos
                    else:
                        past = np.sort(evs[:i+1])              # solo hasta i (causal)
                        thr  = float(np.quantile(past, q))
                        keep.append(evs[i] >= thr)
                        qthr.append(thr)
                q_keep_list.extend(keep)
                q_thr_list.extend(qthr)
            cand["q_keep"] = q_keep_list
            cand["q_thr"]  = q_thr_list

        # === Seleccin causal con cap K/da en ORDEN TEMPORAL ===
        chosen = []
        if not cand.empty:
            for d, g in cand.groupby("day", sort=False):
                g = g[g["q_keep"]].sort_values("open_time")
                if len(g) > 0:
                    chosen.append(g.head(int(max_trades_per_day)))
        chosen = pd.concat(chosen, axis=0) if len(chosen) > 0 else cand.iloc[0:0]
        # chosen_flag por ndice original
        cand["chosen_flag"] = 0
        if not chosen.empty:
            cand.loc[chosen.index, "chosen_flag"] = 1

        # ====== Export candidatos de este slice para auditora T2/T6/T7 ======
        export_cols = [
            "open_time","day","p_up","direction","ev_best","ev_long","ev_shrt","ev_gap",
            "q_thr","q_keep","chosen_flag"
        ]
        cand_exp = cand[export_cols].copy()
        cand_exp["train_start"] = pd.to_datetime(train_start)
        cand_exp["train_end"]   = pd.to_datetime(train_end)
        cand_exp["embargo_sec"] = int(embargo_td.total_seconds())
        cand_exp["bar_seconds"] = int(bar_seconds)
        cand_exports.append(cand_exp)

        # ====== Ejecucin (equity intrada por trade) con RISK FRAMEWORK ======
        bal_before = equity
        trades = wins = losses = 0
        trade_ledger_rows = []

        for _, row in chosen.iterrows():
            if kill_switch_triggered:
                risk_skips += 1
                continue

            _maybe_roll_day(pd.to_datetime(row["open_time"]))
            if equity <= day_floor_equity + 1e-12:
                risk_skips += 1
                days_paused += 1
                continue

            # Sizing base (sin leverage) por volatilidad
            atr_r = float(row["atr"] / max(row["close"], 1e-9))
            s_vol = _vol_position_scale(atr_r, ref_vol)
            stake_unlev = equity * float(position_frac) * s_vol
            if atr_r >= float(extreme_vol):
                stake_unlev *= float(VOL_CUT_FACTOR)
            stake_unlev = min(stake_unlev, equity * float(position_frac_max))
            if stake_unlev <= 0:
                continue

            # Direccin y retorno de peor caso efectivo
            is_long = (row["direction"] == "long")
            ret_sl_eff = float(row["ret_sl_long"] if is_long else row["ret_sl_short"])  # negativo
            fee_pct = float(commission + slippage)

            # Cap de riesgo por presupuesto diario + kill global
            cap_unlev = _stake_cap_from_budget(ret_sl_eff, fee_pct)
            risk_capped = False
            if cap_unlev <= 1e-12:
                risk_skips += 1
                continue
            if stake_unlev > cap_unlev:
                stake_unlev = cap_unlev
                risk_caps_applied += 1
                risk_capped = True

            # Montos con leverage y PnL
            stake_lev = stake_unlev * float(leverage)
            fees = _apply_costs(stake_lev, commission=commission, slippage=slippage)

            # Realiza trade en funcin del label out-of-sample
            if is_long:
                ret = float(row["ret_tp_long"]) if int(row["label"]) == 1 else float(row["ret_sl_long"])
                wins += int(row["label"] == 1); losses += int(row["label"] != 1)
            else:
                ret = float(row["ret_tp_short"]) if int(row["label"]) == 0 else float(row["ret_sl_short"])
                wins += int(row["label"] == 0); losses += int(row["label"] != 0)

            pnl = stake_lev * ret - fees

            # Kill-switch global pre- y post- trade (ajuste fino si aplica)
            worst_loss_if_fills = stake_lev * abs(ret_sl_eff) + fees
            if (equity - worst_loss_if_fills) <= _global_floor() + 1e-12:
                rem_glob = max(0.0, equity - _global_floor())
                tiny = float(leverage) * (abs(ret_sl_eff) + fee_pct)
                if tiny > 0:
                    stake_cap_glob_unlev = rem_glob / tiny
                    if stake_cap_glob_unlev < stake_unlev - 1e-9:
                        stake_unlev = max(0.0, stake_cap_glob_unlev)
                        stake_lev = stake_unlev * float(leverage)
                        fees = _apply_costs(stake_lev, commission=commission, slippage=slippage)
                        pnl = stake_lev * ret - fees
                        risk_caps_applied += 1
                        risk_capped = True

            equity_before = equity
            equity += pnl
            equity_after = equity
            total_fees += float(fees)
            trades += 1
            equity_curve.append({"time": row["open_time"], "equity": equity})

            # ledger row
            trade_ledger_rows.append({
                "time": row["open_time"],
                "direction": row["direction"],
                "p_up": float(row["p_up"]),
                "ev_best": float(row["ev_best"]),
                "stake_unlev": float(stake_unlev),
                "stake_lev": float(stake_lev),
                "fees": float(fees),
                "ret_realized": float(ret),
                "pnl": float(pnl),
                "equity_before": float(equity_before),
                "equity_after": float(equity_after),
                "risk_capped": bool(risk_capped),
            })

            _update_peak(equity)
            if equity <= _global_floor() + 1e-12:
                kill_switch_triggered = True
                print(f"[KILL] Activado MDD global ({mdd_kill_pct*100:.1f}%). Equity={equity:.2f}  Peak={peak_equity:.2f}")

        # Guardar ledger por slice (append al archivo para no reventar memoria)
        if len(trade_ledger_rows) > 0:
            ledger_path = out_dir / "trade_ledger.csv"
            ld = pd.DataFrame(trade_ledger_rows)
            mode = "a" if ledger_path.exists() else "w"
            header = not ledger_path.exists()
            ld.to_csv(ledger_path, index=False, mode=mode, header=header)

        rows.append({
            "train_start": train_start.date(), "train_end": train_end.date(), "test_end": test_end.date(),
            "n_train": len(tr), "n_test": len(te),
            "acc": acc, "precision": p, "recall": r, "f1": f1,
            "equity_start": bal_before, "equity_end": equity,
            "trades": trades, "wins": wins, "losses": losses,
        })

        print(
            f"[SLICE] {train_start.date()}{train_end.date()} | test{test_end.date()} | "
            f"acc={acc:.3f} p={p:.3f} r={r:.3f} f1={f1:.3f} | "
            f"trades={trades} | eq: {bal_before:,.2f}{equity:,.2f} | purged={purged_n}"
        )

        t += step

    # === Resultados / reportes ===
    res = pd.DataFrame(rows)
    eq_intra = pd.DataFrame(equity_curve)

    # Export candidatos (T2/T6/T7)
    if len(cand_exports) > 0:
        D = pd.concat(cand_exports, axis=0, ignore_index=True)
        # Asegura naive/UTC
        D["open_time"]   = pd.to_datetime(D["open_time"]).dt.tz_localize(None)
        D["train_start"] = pd.to_datetime(D["train_start"]).dt.tz_localize(None)
        D["train_end"]   = pd.to_datetime(D["train_end"]).dt.tz_localize(None)
        D.to_csv(out_dir / "walk_candidates.csv", index=False)

    res_path = out_dir / "walk_report.csv"
    res.to_csv(res_path, index=False)
    print(f"\n[RESUMEN] Slices: {len(res)} | Equity final: ${equity:,.2f}")
    print(f"Reporte guardado en: {res_path}")

    # ===== Guardar curva intrada (legacy y estndar) =====
    if not eq_intra.empty:
        eq_intra.to_csv(out_dir / "equity_intraday.csv", index=False)       # legacy
        eq_intra.to_csv(out_dir / "walk_equity_curve.csv", index=False)     # estndar

    # ======= MDD robusto (intrada y diario) + worst steps =======
    def _robust_mdds_from_intraday(intra_df: pd.DataFrame, save_daily_path: Path | None = None):
        if intra_df is None or intra_df.empty:
            if save_daily_path is not None:
                pd.DataFrame({"date": [], "equity": []}).to_csv(save_daily_path, index=False)
            return 0.0, 0.0, 0.0, 0.0, pd.Series(dtype=float)

        s = intra_df.copy()
        s["time"] = pd.to_datetime(s["time"]).dt.tz_localize(None)
        s = s.sort_values("time")
        s = s[~s["time"].duplicated(keep="last")]
        eq = pd.to_numeric(s["equity"], errors="coerce").astype(float)

        # Intrada
        peak = eq.cummax()
        dd = eq / peak - 1.0
        mdd_intra = float(dd.min())
        worst_step = float(eq.pct_change().min())

        # Diario por last-of-day + ffill
        eod = s.set_index("time")["equity"].resample("D").last().ffill()
        peak_d = eod.cummax()
        dd_d = eod / peak_d - 1.0
        mdd_daily = float(dd_d.min())
        worst_daily = float(eod.pct_change().min())

        # Guardar daily curve (estndar)
        if save_daily_path is not None:
            eod.to_frame(name="equity").rename_axis("date").reset_index().to_csv(save_daily_path, index=False)

        return mdd_intra, mdd_daily, worst_step, worst_daily, eod

    mdd_intra, mdd_daily, worst_step, worst_daily, eod_series = _robust_mdds_from_intraday(
        eq_intra, save_daily_path=out_dir / "walk_equity_curve_daily.csv"
    )

    # ======= Comparativa Buy&Hold (opcional, usa cierres diarios) =======
    if not px_daily.empty:
        idx = px_daily.index.union(eod_series.index if eod_series is not None else px_daily.index)
        pxu = px_daily.reindex(idx).ffill()
        robot = (eod_series.reindex(idx).ffill() if isinstance(eod_series, pd.Series) and not eod_series.empty
                 else pd.Series(equity0, index=idx))
        btc_ret = pxu.pct_change().fillna(0.0)
        btc_equity = equity0 * (1.0 + btc_ret).cumprod()
        robot_ret = robot.pct_change().fillna(0.0)
        out = pd.DataFrame({
            "date": idx.values,
            "btc_ret": btc_ret.values,
            "robot_ret": robot_ret.values,
            "btc_cum_ret": (1.0 + btc_ret).cumprod().values - 1.0,
            "robot_cum_ret": (1.0 + robot_ret).cumprod().values - 1.0,
            "btc_equity": btc_equity.values,
            "robot_equity": robot.values,
        })
        out.to_csv(out_dir / "equity_vs_buyhold_daily.csv", index=False)

    # ======= Diagnstico de invariante =======
    print(f"[MDD] intra={mdd_intra:.6f}  daily={mdd_daily:.6f}  worst_step={worst_step:.6f}  worst_daily={worst_daily:.6f}")
    if mdd_intra - 1e-9 > mdd_daily:
        print("[WARN] Invariante rota: intrada > diario (revisa timestamps/tz/ffill).")
    else:
        print("[OK] Invariante consistente: intrada d diario.")

    # ======= CAGR y periodizacin (usando intrada; si vaco, usa diario) =======
    if not eq_intra.empty:
        days = (pd.to_datetime(eq_intra["time"]).iloc[-1] - pd.to_datetime(eq_intra["time"]).iloc[0]).days
    elif isinstance(eod_series, pd.Series) and not eod_series.empty:
        days = int((eod_series.index[-1] - eod_series.index[0]).days)
    else:
        days = 365
    years = max(days / 365.25, 1e-9)
    cagr = (equity / equity0) ** (1 / years) - 1 if equity > 0 else -1.0

    # ======= Summary JSON (usa mtricas robustas + risk stats) =======
    summary = {
        "dataset_path": str(data_path),
        "dataset_sha256": sha,
        "dataset_rows": int(n_rows),
        "equity_final": float(equity),
        "slices": int(len(res)),
        "trades_total": int(res["trades"].sum()) if not res.empty else 0,
        "win_rate": (float(res["wins"].sum()) / max(1, float(res["trades"].sum()))) if not res.empty else 0.0,
        "f1_mean": float(res["f1"].mean()) if not res.empty else 0.0,
        "precision_mean": float(res["precision"].mean()) if not res.empty else 0.0,
        "recall_mean": float(res["recall"].mean()) if not res.empty else 0.0,
        "acc_mean": float(res["acc"].mean()) if not res.empty else 0.0,
        # Mtricas robustas
        "max_drawdown_intraday": float(mdd_intra),
        "max_drawdown_daily": float(mdd_daily),
        "worst_step_return_intraday": float(worst_step),
        "worst_daily_return": float(worst_daily),
        "total_fees": float(total_fees),
        "CAGR": float(cagr),
        # Estado de riesgo
        "risk_stats": {
            "risk_daily_pct": float(risk_daily_pct),
            "mdd_kill_pct": float(mdd_kill_pct),
            "daily_budget_is_net": bool(daily_budget_is_net),
            "risk_caps_applied": int(risk_caps_applied),
            "risk_skips": int(risk_skips),
            "days_paused": int(days_paused),
            "kill_switch_triggered": bool(kill_switch_triggered),
        },
        "params": {
            "window_days": window_days, "step_days": step_days, "TP_MULT": TP_MULT, "SL_MULT": SL_MULT, "HORIZON": HORIZON,
            "commission": commission, "slippage": slippage, "position_frac": position_frac, "position_frac_max": position_frac_max,
            "ev_min_ret": ev_min_ret, "ev_min_perc_stake": ev_min_perc_stake, "prob_edge_min": prob_edge_min,
            "ev_gap_perc": ev_gap_perc, "daily_ev_quantile": daily_ev_quantile, "min_obs_for_q": min_obs_for_q, "leverage": leverage
        },
    }

    with open(out_dir / "walk_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[SUMMARY]", json.dumps(summary, indent=2, default=str))

    return res

if __name__ == "__main__":
    run_walk()
