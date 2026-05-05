"""
RoboTrader S4 — data_fetcher.py
OKX BTC-USDT-SWAP — sin restricciones de IP desde cualquier datacenter.
Precios identicos a Binance Futures (arbitraje <0.1%).
"""
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config import INTERVAL

_BASE = "https://www.okx.com"
_INST = "BTC-USDT-SWAP"

_INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D",
}

def get_klines(symbol: str, interval: str, limit: int = 100,
               after: Optional[int] = None) -> list:
    bar = _INTERVAL_MAP.get(interval)
    if not bar:
        raise ValueError(f"Intervalo no soportado: {interval}")
    params = {"instId": _INST, "bar": bar, "limit": min(limit, 100)}
    if after:
        params["after"] = after
    r = requests.get(_BASE + "/api/v5/market/candles",
                     params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX error: {data.get('msg')}")
    return data["data"]  # [[ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm], ...]

def klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close",
        "volume","volCcy","volCcyQuote","confirm"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"]), unit="ms", utc=True
    )
    df["close_time"] = df["open_time"] + pd.Timedelta(hours=1)
    df = df[["open_time","open","high","low","close","volume","close_time"]]
    # Filtrar solo velas confirmadas (confirm=1) y ordenar
    if "confirm" in df.columns:
        df = df[df["confirm"].astype(str) == "1"]
    return df.sort_values("open_time").reset_index(drop=True)

def get_historical_data(symbol: str = "BTCUSDT", interval: str = INTERVAL,
                        limit: int = 300, start=None, end=None,
                        sleep_sec: float = 0.3) -> pd.DataFrame:
    """
    Descarga las ultimas `limit` velas de OKX.
    OKX devuelve max 100 por llamada — pagina automaticamente.
    """
    frames = []
    remaining = limit
    after = None

    while remaining > 0:
        batch = min(100, remaining)
        kl = get_klines(symbol, interval, limit=batch, after=after)
        if not kl:
            break
        df = klines_to_df(kl)
        if df.empty:
            break
        frames.append(df)
        remaining -= len(df)
        if len(df) < batch:
            break
        # Paginar: after = timestamp mas antiguo - 1ms
        after = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return out.iloc[-limit:].reset_index(drop=True)

if __name__ == "__main__":
    df = get_historical_data("BTCUSDT", "1h", limit=5)
    print(df[["open_time","open","high","low","close","volume"]])
