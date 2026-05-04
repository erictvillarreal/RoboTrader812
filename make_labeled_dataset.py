"""
RoboTrader S4 — make_labeled_dataset.py
Etiqueta el CSV raw con Triple Barrera (causal, sin look-ahead).
Guarda LABELED_CSV listo para walk.py.

Uso:
    python make_labeled_dataset.py
    python make_labeled_dataset.py --tp 2.0 --sl 0.8 --H 12
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import RAW_CSV, LABELED_CSV, FEATURES, TP_MULT, SL_MULT, HORIZON
from tech_signals import add_technical_signals

# ── Utilidades ────────────────────────────────────────────

def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalizar tiempo
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"[load] {len(df)} filas desde {path}")
    return df

# ── Triple Barrera (causal) ───────────────────────────────

def triple_barrier(df: pd.DataFrame, tp_mult: float, sl_mult: float,
                   H: int) -> pd.DataFrame:
    """
    label = 1  → TP tocado primero dentro de H velas
    label = 0  → SL tocado primero dentro de H velas
    Filas donde no se resuelve en H velas → descartadas (sin look-ahead).

    Incluye columnas auxiliares para walk.py:
        ret_tp_long, ret_sl_long, ret_tp_short, ret_sl_short
    """
    df = df.copy().reset_index(drop=True)
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr   = df["atr"].to_numpy()

    labels, keep = np.full(len(df), -1, dtype=np.int8), np.zeros(len(df), dtype=bool)

    for i in range(len(df) - H):
        entry = close[i]
        a     = atr[i]
        if entry <= 0 or a <= 0:
            continue
        tp = entry + tp_mult * a
        sl = entry - sl_mult * a

        for j in range(i + 1, i + H + 1):
            if high[j] >= tp:
                labels[i] = 1; keep[i] = True; break
            if low[j]  <= sl:
                labels[i] = 0; keep[i] = True; break

    df["label"] = labels
    df = df[keep].copy().reset_index(drop=True)

    # Retornos teóricos hasta barreras (para EV en walk.py)
    df["ret_tp_long"]  =  tp_mult * df["atr"] / df["close"]
    df["ret_sl_long"]  = -sl_mult * df["atr"] / df["close"]
    df["ret_tp_short"] =  tp_mult * df["atr"] / df["close"]   # short: gana si baja
    df["ret_sl_short"] = -sl_mult * df["atr"] / df["close"]

    for c in ["ret_tp_long","ret_sl_long","ret_tp_short","ret_sl_short"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df

# ── Main ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp",  type=float, default=TP_MULT)
    ap.add_argument("--sl",  type=float, default=SL_MULT)
    ap.add_argument("--H",   type=int,   default=HORIZON)
    ap.add_argument("--in",  dest="inp", default=str(RAW_CSV))
    ap.add_argument("--out", default=str(LABELED_CSV))
    args = ap.parse_args()

    inp_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not inp_path.exists():
        raise FileNotFoundError(f"No existe el CSV raw: {inp_path}\n"
                                f"Corre primero: python update_data.py")

    df = _load_raw(inp_path)
    df = add_technical_signals(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"[signals] {len(df)} filas con features completas")

    df_l = triple_barrier(df, tp_mult=args.tp, sl_mult=args.sl, H=args.H)
    wins    = (df_l["label"] == 1).sum()
    losses  = (df_l["label"] == 0).sum()
    print(f"[labels] {len(df_l)} etiquetadas | wins={wins} ({wins/len(df_l):.1%}) | losses={losses}")

    # Verificar que todas las features existan
    missing = [c for c in FEATURES if c not in df_l.columns]
    if missing:
        raise ValueError(f"Faltan features: {missing}")

    df_l.to_csv(out_path, index=False)
    print(f"[OK] Guardado: {out_path} ({len(df_l)} filas)")
    return df_l

if __name__ == "__main__":
    main()
