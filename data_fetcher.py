"""
RoboTrader S4 — data_fetcher.py
Bybit Linear Perpetuals (BTCUSDT) — sin restricciones de IP.
Mismos datos OHLCV que Binance Futures, compatible con el modelo S4.
"""
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config import INTERVAL

_BASE = "https://api.bybit.com"

_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D",
}

def get_klines(symbol: str, interval: str, limit: int = 200,
               end_ms: Optional[int] = None) -> list:
    iv = _INTERVAL_MAP.get(interval)
    if not iv:
        raise ValueError(f"Intervalo no soportado: {interval}")
    params = {
        "category": "linear",
        "symbol":   symbol,
        "interval": iv,
        "limit":    min(limit, 200),
    }
    if end_ms:
        params["end"] = end_ms
    r = requests.get(_BASE + "/v5/market/kline", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
    return data["result"]["list"]   # [[ts, open, high, low, close, vol, turnover], ...]

def klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume", "turnover"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"]), unit="ms", utc=True
    )
    df["close_time"] = df["open_time"] + pd.Timedelta(hours=1)
    df = df[["open_time","open","high","low","close","volume","close_time"]]
    return df.sort_values("open_time").reset_index(drop=True)

def get_historical_data(symbol: str = "BTCUSDT", interval: str = INTERVAL,
                        limit: int = 300, start=None, end=None,
                        sleep_sec: float = 0.3) -> pd.DataFrame:
    """
    Descarga las ultimas `limit` velas de Bybit.
    Bybit devuelve max 200 por llamada — pagina automaticamente.
    """
    frames = []
    remaining = limit
    end_ms = None

    while remaining > 0:
        batch = min(200, remaining)
        kl = get_klines(symbol, interval, limit=batch, end_ms=end_ms)
        if not kl:
            break
        df = klines_to_df(kl)
        if df.empty:
            break
        frames.append(df)
        remaining -= len(df)
        if remaining <= 0 or len(df) < batch:
            break
        # Paginar hacia atras
        end_ms = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return out.iloc[-limit:].reset_index(drop=True)

if __name__ == "__main__":
    df = get_historical_data("BTCUSDT", "1h", limit=5)
    print(df[["open_time","open","high","low","close","volume"]])
