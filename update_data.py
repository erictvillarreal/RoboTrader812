"""
RoboTrader S4 — update_data.py
Descarga/actualiza el CSV raw de BTCUSDT (Binance Futures).
Primera corrida: descarga ~3 años. Corridas siguientes: solo lo nuevo.

Uso:
    python update_data.py
    python update_data.py --symbol BTCUSDT --interval 1h --limit 26000
"""
import argparse
import pandas as pd
from pathlib import Path

from config import RAW_CSV, SYMBOL, INTERVAL
from data_fetcher import get_historical_data

def update_csv(symbol: str, interval: str, limit: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Carga existente
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
        print(f"[load] {len(existing)} filas existentes en {out_path}")
    else:
        existing = pd.DataFrame()
        print(f"[load] Primera descarga — solicitando {limit} velas")

    # Descarga
    print(f"[fetch] {symbol} {interval} — solicitando hasta {limit} velas...")
    new_df = get_historical_data(symbol, interval, limit=limit)
    if new_df.empty:
        print("[fetch] Sin datos. Verifica conectividad o USE_TESTNET.")
        return

    print(f"[fetch] {len(new_df)} velas descargadas")

    # Merge + dedup
    if not existing.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    merged = (merged
              .drop_duplicates("open_time")
              .sort_values("open_time")
              .reset_index(drop=True))

    merged.to_csv(out_path, index=False)
    print(f"[OK] {out_path} — {len(merged)} filas totales | "
          f"desde {merged['open_time'].iloc[0]} "
          f"hasta {merged['open_time'].iloc[-1]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",   default=SYMBOL)
    ap.add_argument("--interval", default=INTERVAL)
    ap.add_argument("--limit",    type=int, default=26000)
    ap.add_argument("--out",      default=str(RAW_CSV))
    args = ap.parse_args()
    update_csv(args.symbol, args.interval, args.limit, Path(args.out))

if __name__ == "__main__":
    main()
