"""Fetches market data for one instrument group (gold or crypto) from Twelve
Data, computes indicators, builds a trade setup, and merges the result into
data/latest.json - preserving whatever the OTHER group last wrote, since gold
and crypto run on separate schedules and both touch this same file. Notifies
via ntfy.sh only when a real BUY/SELL setup forms (not on reverting to HOLD).
Standalone - no MT5, safe for GitHub Actions.

Usage: python fetch_market_data.py <gold|crypto>
"""
import json
import os
import sys
from decimal import Decimal
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from indicators import (
    arnaud_legoux_moving_average,
    average_directional_index,
    average_true_range,
    moving_average_convergence_divergence,
    relative_strength_index,
    rsi_moving_average,
)

load_dotenv()

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

GROUPS = {
    "gold": {"XAU/USD": "5min"},
    "crypto": {"BTC/USD": "15min", "ETH/USD": "15min", "SOL/USD": "15min"},
}

OUTPUTSIZE = 50
DATA_PATH = "data/latest.json"

STOP_ATR_MULT = Decimal("1.5")
TARGET_ATR_MULT = Decimal("3.0")

BASE_URL = "https://api.twelvedata.com/time_series"


def fetch_symbol(symbol: str, interval: str) -> dict:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": OUTPUTSIZE,
        "apikey": API_KEY,
    }
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if data.get("status") == "error":
        raise RuntimeError(f"{symbol}: {data.get('message')}")
    return data


def to_num(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def compute_signal(
    close: Decimal, alma: Decimal, rsi: Decimal, rsi_ma: Decimal, macd: Decimal, macd_signal: Decimal
) -> str:
    if close > alma and rsi > rsi_ma and macd > macd_signal:
        return "BUY"
    if close < alma:
        return "SELL"
    return "HOLD"


def determine_regime(adx_value: Decimal | None) -> str | None:
    if adx_value is None:
        return None
    if adx_value >= Decimal(25):
        return "trending"
    if adx_value < Decimal(20):
        return "ranging"
    return "transitional"


def build_trade_setup(signal: str | None, price: Decimal, atr: Decimal | None) -> dict:
    if signal is None or signal == "HOLD" or atr is None:
        return {"entry": None, "stop_loss": None, "take_profit": None, "risk_reward": None}
    if signal == "BUY":
        stop = price - STOP_ATR_MULT * atr
        target = price + TARGET_ATR_MULT * atr
        risk = price - stop
        reward = target - price
    else:
        stop = price + STOP_ATR_MULT * atr
        target = price - TARGET_ATR_MULT * atr
        risk = stop - price
        reward = price - target
    rr = (reward / risk) if risk > 0 else None
    return {
        "entry": to_num(price),
        "stop_loss": to_num(stop),
        "take_profit": to_num(target),
        "risk_reward": to_num(rr),
    }


def load_existing_data() -> dict:
    if not os.path.exists(DATA_PATH):
        return {"updated_at": None, "instruments": {}}
    try:
        with open(DATA_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {"updated_at": None, "instruments": {}}


def send_notification(title: str, message: str) -> None:
    try:
        requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
        print(f"  Notified: {title} - {message}")
    except requests.RequestException as exc:
        print(f"  Notification failed (non-fatal): {exc}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in GROUPS:
        raise SystemExit(f"Usage: python fetch_market_data.py <{'|'.join(GROUPS)}>")
    group_name = sys.argv[1]
    instruments = GROUPS[group_name]

    output = load_existing_data()
    output.setdefault("instruments", {})
    previous_signals = {
        symbol: info.get("signal") for symbol, info in output["instruments"].items()
    }

    for symbol, interval in instruments.items():
        print(f"Fetching {symbol} ({interval})...")
        data = fetch_symbol(symbol, interval)
        candles = list(reversed(data["values"]))
        closes = [Decimal(c["close"]) for c in candles]
        highs = [Decimal(c["high"]) for c in candles]
        lows = [Decimal(c["low"]) for c in candles]

        rsi = relative_strength_index(closes)
        rsi_ma = rsi_moving_average(closes)
        macd_result = moving_average_convergence_divergence(closes)
        alma = arnaud_legoux_moving_average(closes)
        adx = average_directional_index(highs, lows, closes)
        atr = average_true_range(highs, lows, closes)

        latest_idx = len(closes) - 1
        price = closes[latest_idx]

        signal = None
        if all(
            v[latest_idx] is not None
            for v in (alma, rsi, rsi_ma, macd_result.macd, macd_result.signal)
        ):
            signal = compute_signal(
                price,
                alma[latest_idx],
                rsi[latest_idx],
                rsi_ma[latest_idx],
                macd_result.macd[latest_idx],
                macd_result.signal[latest_idx],
            )

        regime = determine_regime(adx[latest_idx])
        trade_setup = build_trade_setup(signal, price, atr[latest_idx])

        output["instruments"][symbol] = {
            "interval": interval,
            "meta": data["meta"],
            "latest_close": to_num(price),
            "latest_datetime": candles[latest_idx]["datetime"],
            "signal": signal,
            "regime": regime,
            "trade_setup": trade_setup,
            "indicators": {
                "rsi": to_num(rsi[latest_idx]),
                "rsi_ma": to_num(rsi_ma[latest_idx]),
                "macd": to_num(macd_result.macd[latest_idx]),
                "macd_signal": to_num(macd_result.signal[latest_idx]),
                "macd_histogram": to_num(macd_result.histogram[latest_idx]),
                "alma": to_num(alma[latest_idx]),
                "adx": to_num(adx[latest_idx]),
                "atr": to_num(atr[latest_idx]),
            },
            "candles": [
                {
                    "datetime": c["datetime"],
                    "close": to_num(closes[i]),
                    "rsi": to_num(rsi[i]),
                    "rsi_ma": to_num(rsi_ma[i]),
                    "macd": to_num(macd_result.macd[i]),
                    "macd_signal": to_num(macd_result.signal[i]),
                    "alma": to_num(alma[i]),
                    "adx": to_num(adx[i]),
                }
                for i, c in enumerate(candles)
            ],
        }

        print(
            f"  Latest close: {price}  Signal: {signal}  Regime: {regime}  "
            f"Setup: {trade_setup}"
        )

        old_signal = previous_signals.get(symbol)
        symbol_is_first_run = symbol not in previous_signals
        is_new_setup = signal in ("BUY", "SELL") and signal != old_signal
        if not symbol_is_first_run and is_new_setup and trade_setup["entry"] is not None:
            body = (
                f"{signal} setup ({regime or 'unknown regime'})\n"
                f"Entry: {trade_setup['entry']:.2f}\n"
                f"SL: {trade_setup['stop_loss']:.2f}\n"
                f"TP: {trade_setup['take_profit']:.2f}\n"
                f"R:R {trade_setup['risk_reward']:.1f}"
            )
            send_notification(title=f"{symbol}: {signal} setup", message=body)

    output["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(DATA_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {DATA_PATH} (group: {group_name})")


if __name__ == "__main__":
    main()