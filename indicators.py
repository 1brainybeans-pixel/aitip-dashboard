"""Standalone indicator calculations, ported from the aitip package.
No aitip import, no MT5 anywhere in the chain - safe to run in GitHub Actions.
Math is unchanged from the validated originals; adaptations:
1. IndicatorError is now a plain local exception (was aitip.core.exceptions).
2. on_balance_volume takes two parallel lists (closes, volumes) instead of a
   Candle object, since there's no Candle class in this repo.
3. average_directional_index takes three parallel lists (highs, lows, closes)
   for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import exp
from typing import Sequence


class IndicatorError(ValueError):
    """Raised when indicator inputs are invalid."""


_HUNDRED = Decimal(100)


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def simple_moving_average(
    prices: Sequence[Decimal], period: int
) -> list[Decimal | None]:
    if period <= 0:
        raise IndicatorError("period must be a positive integer")
    result: list[Decimal | None] = []
    window_sum = Decimal(0)
    for i, price in enumerate(prices):
        window_sum += price
        if i >= period:
            window_sum -= prices[i - period]
        if i >= period - 1:
            result.append(window_sum / period)
        else:
            result.append(None)
    return result


def exponential_moving_average(
    prices: Sequence[Decimal], period: int
) -> list[Decimal | None]:
    if period <= 0:
        raise IndicatorError("period must be a positive integer")
    result: list[Decimal | None] = []
    k = Decimal(2) / Decimal(period + 1)
    prev_ema: Decimal | None = None
    window_sum = Decimal(0)
    for i, price in enumerate(prices):
        if i < period - 1:
            window_sum += price
            result.append(None)
            continue
        if prev_ema is None:
            window_sum += price
            prev_ema = window_sum / period
        else:
            prev_ema = price * k + prev_ema * (1 - k)
        result.append(prev_ema)
    return result


# ---------------------------------------------------------------------------
# RSI and RSI-MA
# ---------------------------------------------------------------------------

def relative_strength_index(
    prices: Sequence[Decimal], period: int = 14
) -> list[Decimal | None]:
    if period <= 0:
        raise IndicatorError("period must be a positive integer")
    result: list[Decimal | None] = [None] * len(prices)
    if len(prices) <= period:
        return result
    total_gain = Decimal(0)
    total_loss = Decimal(0)
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            total_gain += change
        else:
            total_loss += -change
    avg_gain = total_gain / period
    avg_loss = total_loss / period
    result[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = change if change > 0 else Decimal(0)
        loss = -change if change < 0 else Decimal(0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        result[i] = _rsi_value(avg_gain, avg_loss)
    return result


def _rsi_value(avg_gain: Decimal, avg_loss: Decimal) -> Decimal:
    if avg_loss == 0:
        return _HUNDRED
    rs = avg_gain / avg_loss
    return _HUNDRED - (_HUNDRED / (1 + rs))


class MaType(StrEnum):
    SMA = "sma"
    EMA = "ema"


def rsi_moving_average(
    prices: Sequence[Decimal],
    rsi_period: int = 14,
    ma_period: int = 14,
    ma_type: MaType = MaType.SMA,
) -> list[Decimal | None]:
    if ma_period <= 0:
        raise IndicatorError("ma_period must be a positive integer")
    rsi = relative_strength_index(prices, rsi_period)
    real: list[Decimal] = []
    first_real_index: int | None = None
    for i, value in enumerate(rsi):
        if value is not None:
            if first_real_index is None:
                first_real_index = i
            real.append(value)
    result: list[Decimal | None] = [None] * len(prices)
    if not real or first_real_index is None:
        return result
    if ma_type is MaType.SMA:
        smoothed = simple_moving_average(real, ma_period)
    else:
        smoothed = exponential_moving_average(real, ma_period)
    for offset, value in enumerate(smoothed):
        result[first_real_index + offset] = value
    return result


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class MacdMaType(StrEnum):
    EMA = "ema"
    SMA = "sma"


@dataclass(frozen=True)
class MacdResult:
    macd: list[Decimal | None]
    signal: list[Decimal | None]
    histogram: list[Decimal | None]


def _ma(
    prices: Sequence[Decimal], period: int, ma_type: MacdMaType
) -> list[Decimal | None]:
    if ma_type is MacdMaType.EMA:
        return exponential_moving_average(prices, period)
    return simple_moving_average(prices, period)


def moving_average_convergence_divergence(
    prices: Sequence[Decimal],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    osc_ma_type: MacdMaType = MacdMaType.EMA,
    signal_ma_type: MacdMaType = MacdMaType.EMA,
) -> MacdResult:
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise IndicatorError("all periods must be positive integers")
    if fast_period >= slow_period:
        raise IndicatorError("fast_period must be smaller than slow_period")
    n = len(prices)
    fast = _ma(prices, fast_period, osc_ma_type)
    slow = _ma(prices, slow_period, osc_ma_type)
    macd: list[Decimal | None] = [None] * n
    for i in range(n):
        f = fast[i]
        s = slow[i]
        if f is not None and s is not None:
            macd[i] = f - s
    real: list[Decimal] = []
    first: int | None = None
    for i, v in enumerate(macd):
        if v is not None:
            if first is None:
                first = i
            real.append(v)
    signal: list[Decimal | None] = [None] * n
    if real and first is not None:
        smoothed = _ma(real, signal_period, signal_ma_type)
        for offset, v in enumerate(smoothed):
            signal[first + offset] = v
    histogram: list[Decimal | None] = [None] * n
    for i in range(n):
        m = macd[i]
        sig = signal[i]
        if m is not None and sig is not None:
            histogram[i] = m - sig
    return MacdResult(macd=macd, signal=signal, histogram=histogram)


# ---------------------------------------------------------------------------
# OBV - adapted signature: (closes, volumes) instead of Sequence[Candle]
# ---------------------------------------------------------------------------

def on_balance_volume(
    closes: Sequence[Decimal], volumes: Sequence[Decimal]
) -> list[Decimal | None]:
    if len(closes) != len(volumes):
        raise IndicatorError("closes and volumes must be the same length")
    result: list[Decimal | None] = []
    if not closes:
        return result
    result.append(None)
    running = Decimal(0)
    for i in range(1, len(closes)):
        close = closes[i]
        prev_close = closes[i - 1]
        volume = volumes[i]
        if close > prev_close:
            running += volume
        elif close < prev_close:
            running -= volume
        result.append(running)
    return result


# ---------------------------------------------------------------------------
# ALMA
# ---------------------------------------------------------------------------

def arnaud_legoux_moving_average(
    prices: Sequence[Decimal],
    period: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> list[Decimal | None]:
    if period <= 0:
        raise IndicatorError("period must be a positive integer")
    if not 0.0 <= offset <= 1.0:
        raise IndicatorError("offset must be between 0 and 1")
    if sigma <= 0:
        raise IndicatorError("sigma must be positive")
    result: list[Decimal | None] = [None] * len(prices)
    if len(prices) < period:
        return result
    m = offset * (period - 1)
    s = period / sigma
    raw = [Decimal(str(exp(-((i - m) ** 2) / (2 * s * s)))) for i in range(period)]
    total = sum(raw, Decimal(0))
    if total == 0:
        raise IndicatorError("degenerate ALMA weights")
    weights = [w / total for w in raw]
    for end in range(period - 1, len(prices)):
        window = prices[end - period + 1 : end + 1]
        acc = Decimal(0)
        for price, weight in zip(window, weights, strict=True):
            acc += price * weight
        result[end] = acc
    return result


# ---------------------------------------------------------------------------
# ADX - adapted signature: (highs, lows, closes) instead of Sequence[Candle]
# ---------------------------------------------------------------------------

def average_directional_index(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    period: int = 14,
) -> list[Decimal | None]:
    """Trend-strength gauge, 0-100. Below ~20: ranging. Above ~25: trending.
    Says nothing about direction, only strength. Needs roughly 2*period bars
    to warm up.
    """
    if period <= 0:
        raise IndicatorError("period must be a positive integer")
    if not (len(highs) == len(lows) == len(closes)):
        raise IndicatorError("highs, lows, and closes must be the same length")
    n = len(closes)
    result: list[Decimal | None] = [None] * n
    if n <= 2 * period:
        return result

    plus_dm: list[Decimal] = []
    minus_dm: list[Decimal] = []
    true_range: list[Decimal] = []
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else Decimal(0))
        minus_dm.append(
            down_move if down_move > up_move and down_move > 0 else Decimal(0)
        )
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_range.append(max(high_low, high_close, low_close))

    def _wilder(values: list[Decimal]) -> list[Decimal]:
        smoothed: list[Decimal] = []
        running = sum(values[:period], Decimal(0))
        smoothed.append(running)
        for i in range(period, len(values)):
            running = running - (running / period) + values[i]
            smoothed.append(running)
        return smoothed

    tr_s = _wilder(true_range)
    plus_s = _wilder(plus_dm)
    minus_s = _wilder(minus_dm)

    dx: list[Decimal] = []
    for tr, pdm, mdm in zip(tr_s, plus_s, minus_s, strict=True):
        if tr == 0:
            dx.append(Decimal(0))
            continue
        plus_di = _HUNDRED * pdm / tr
        minus_di = _HUNDRED * mdm / tr
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx.append(Decimal(0))
        else:
            dx.append(_HUNDRED * abs(plus_di - minus_di) / di_sum)

    if len(dx) < period:
        return result
    adx = sum(dx[:period], Decimal(0)) / period
    first_adx_index = 2 * period
    if first_adx_index < n:
        result[first_adx_index] = adx
    for k in range(period, len(dx)):
        adx = (adx * (period - 1) + dx[k]) / period
        idx = period + 1 + k
        if idx < n:
            result[idx] = adx
    return result

# ---------------------------------------------------------------------------
# ATR - adapted signature: (highs, lows, closes) instead of Sequence[Candle]
# ---------------------------------------------------------------------------

def average_true_range(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    period: int = 14,
) -> list[Decimal | None]:
    """Volatility measure, in the instrument's own price units. Used to scale
    stop-loss/take-profit distances so they widen in volatile markets and
    tighten in calm ones.
    """
    if period <= 0:
        raise IndicatorError("period must be a positive integer")
    if not (len(highs) == len(lows) == len(closes)):
        raise IndicatorError("highs, lows, and closes must be the same length")
    n = len(closes)
    result: list[Decimal | None] = [None] * n
    if n <= period:
        return result

    true_ranges: list[Decimal] = []
    for i in range(1, n):
        prev_close = closes[i - 1]
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - prev_close)
        low_close = abs(lows[i] - prev_close)
        true_ranges.append(max(high_low, high_close, low_close))

    seed = sum(true_ranges[:period], Decimal(0)) / period
    result[period] = seed

    atr = seed
    for i in range(period + 1, n):
        tr = true_ranges[i - 1]
        atr = (atr * (period - 1) + tr) / period
        result[i] = atr

    return result