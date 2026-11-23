"""
patterns.py
Rule-based chart-pattern tags: Volatility Contraction Pattern (VCP) and Bull Flag.

TAG-ONLY (1.9.0). Nothing here feeds a hard gate, the soft score or
SCORE_VERSION. Tags are published per row (and stamped into snapshots) so the
outcome tracker can measure them first. AGENTS.md is explicit that a component
is not weighted on in-sample evidence; these have none yet.

Expect tags mostly on rows that FAIL the gates. The gates select names 15-40%
off their 52-week high with RSI 25-40 (a pullback screen); a VCP or a flag
forms near highs (a momentum setup). The tags are a second lens on the same
universe, not a modifier of the first.

Rules follow the published spec of the swing-alpha project
(github.com/Aman123axr/swing-alpha, README "Pattern Detection"). This is an
independent implementation written from that description; that repository
carries no licence, so no code was copied from it. Where the spec is silent the
choice made here is documented inline.

Every detector is pure (DataFrame in, dict-or-None out) and detect_patterns()
never raises: a pattern bug must not cost a row its technicals.
"""

from typing import Optional

import numpy as np
import pandas as pd

from settings import (
    PATTERN_FLAG_MAX_BARS,
    PATTERN_FLAG_MAX_CLOSES_BELOW_EMA20,
    PATTERN_FLAG_MAX_RETRACE_PCT,
    PATTERN_FLAG_MAX_SLOPE_PCT_PER_BAR,
    PATTERN_FLAG_MIN_BARS,
    PATTERN_FLAG_POLE_MAX_BARS,
    PATTERN_FLAG_POLE_MIN_BARS,
    PATTERN_FLAG_POLE_MIN_GAIN_PCT,
    PATTERN_PIVOT_PROXIMITY_PCT,
    PATTERN_SWING_WINDOW,
    PATTERN_VCP_FINAL_DEPTH_MAX_PCT,
    PATTERN_VCP_LOOKBACK,
    PATTERN_VCP_MIN_CONTRACTIONS,
    PATTERN_VCP_PIVOT_MAX_BELOW_HIGH_PCT,
    PATTERN_VCP_VOL_DRYUP_MAX,
)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _clean(hist: pd.DataFrame) -> pd.DataFrame:
    """OHLCV rows with no NaN, in the original order."""
    return hist[OHLCV].dropna()


def pivot_state(close: float, pivot: float) -> str:
    """forming | at_pivot | breakout, relative to a resistance level."""
    if close > pivot:
        return "breakout"
    if close >= pivot * (1 - PATTERN_PIVOT_PROXIMITY_PCT / 100):
        return "at_pivot"
    return "forming"


def swing_high_indices(high: np.ndarray, window: int = PATTERN_SWING_WINDOW) -> list:
    """
    Index i is a swing high when high[i] is the maximum of high[i-w .. i+w] and
    strictly above every bar to its left in that span (so a flat top of equal
    highs yields one swing point, its first bar). The last `w` bars cannot be
    confirmed and are never swing points.
    """
    n = len(high)
    out = []
    for i in range(window, n - window):
        left = high[i - window:i]
        right = high[i + 1:i + window + 1]
        if high[i] > left.max() and high[i] >= right.max():
            out.append(i)
    return out


def detect_vcp(hist: pd.DataFrame) -> Optional[dict]:
    """
    Volatility Contraction Pattern over the last PATTERN_VCP_LOOKBACK sessions.

    1. Swing highs within +/-PATTERN_SWING_WINDOW bars.
    2. Each swing high opens a contraction that runs to the next swing high
       (the last one runs to today). Its depth is the fall from that high to
       the lowest low inside the contraction. Spec says "pair each swing high
       with the next swing low"; the lowest low between highs is the same
       thing without depending on a separately confirmed swing low.
    3. The trailing run of strictly shrinking depths, ending with the current
       contraction, must have >= PATTERN_VCP_MIN_CONTRACTIONS legs.
    4. Average volume of the current leg < first leg of the run x DRYUP_MAX.
    5. Pivot (the last swing high) within PIVOT_MAX_BELOW_HIGH_PCT of the
       lookback high.
    6. Current leg shallower than FINAL_DEPTH_MAX_PCT.
    State (forming / at_pivot / breakout) is reported, not required.
    """
    df = _clean(hist).tail(PATTERN_VCP_LOOKBACK)
    if len(df) < PATTERN_VCP_LOOKBACK:
        return None
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    close = float(df["Close"].iloc[-1])
    n = len(df)

    highs = swing_high_indices(high)
    if len(highs) < PATTERN_VCP_MIN_CONTRACTIONS:
        return None

    legs = []
    for k, start in enumerate(highs):
        end = highs[k + 1] if k + 1 < len(highs) else n  # exclusive
        trough = float(low[start:end].min())
        legs.append({
            "start": start,
            "end": end,
            "high": float(high[start]),
            "depth_pct": float((high[start] - trough) / high[start] * 100),
            "avg_vol": float(vol[start:end].mean()),
        })

    run = [legs[-1]]
    for leg in reversed(legs[:-1]):
        if leg["depth_pct"] > run[0]["depth_pct"]:
            run.insert(0, leg)
        else:
            break
    if len(run) < PATTERN_VCP_MIN_CONTRACTIONS:
        return None

    first, final = run[0], run[-1]
    if first["avg_vol"] <= 0:
        return None
    vol_dryup = float(final["avg_vol"] / first["avg_vol"])
    if vol_dryup >= PATTERN_VCP_VOL_DRYUP_MAX:
        return None

    pivot = final["high"]
    lookback_high = float(high.max())
    pivot_below_high_pct = (lookback_high - pivot) / lookback_high * 100
    if pivot_below_high_pct > PATTERN_VCP_PIVOT_MAX_BELOW_HIGH_PCT:
        return None
    if final["depth_pct"] >= PATTERN_VCP_FINAL_DEPTH_MAX_PCT:
        return None

    return {
        "contractions": len(run),
        "depths_pct": [round(leg["depth_pct"], 1) for leg in run],
        "final_depth_pct": round(final["depth_pct"], 2),
        "vol_dryup_ratio": round(vol_dryup, 2),
        "pivot": round(pivot, 2),
        "pivot_below_high_pct": round(pivot_below_high_pct, 2),
        "pct_to_pivot": round((close / pivot - 1) * 100, 2),
        "state": pivot_state(close, pivot),
        "base_sessions": n - first["start"],
    }


def detect_bull_flag(hist: pd.DataFrame) -> Optional[dict]:
    """
    Bull flag ending today. Searches pole tops from the most recent valid one.

    For a pole top at bar t, the flag is every bar after t (spec rule 2).
    Today's bar is the trigger bar: it may break out, so the flag's own
    rules (stay under the pole high, volume, slope, pivot) use the flag
    bars before today; the closes-below-EMA20 count and the retracement use
    every flag bar, today included, so a collapse today still disqualifies.

    1. Pole: L in [POLE_MIN_BARS, POLE_MAX_BARS] sessions ending at t, with
       close-to-close gain >= POLE_MIN_GAIN_PCT and bar t the pole's highest
       high. The largest-gain L is used.
    2. Flag bars before today stay at or under the pole high.
    3. Flag average volume < pole average volume.
    4. At most MAX_CLOSES_BELOW_EMA20 flag closes below the 20 EMA.
    5. Regression slope of flag closes <= MAX_SLOPE_PCT_PER_BAR per bar.
    6. Retracement of the pole height <= MAX_RETRACE_PCT.
    Pivot = the flag's high before today; state is reported, not required.
    """
    df = _clean(hist)
    need = PATTERN_FLAG_POLE_MAX_BARS + PATTERN_FLAG_MAX_BARS + 21
    if len(df) < need:
        return None
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close_arr = df["Close"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    ema20 = df["Close"].ewm(span=20, adjust=False).mean().to_numpy(dtype=float)
    n = len(df)
    today = n - 1
    close = float(close_arr[today])

    for flag_bars in range(PATTERN_FLAG_MIN_BARS, PATTERN_FLAG_MAX_BARS + 1):
        t = today - flag_bars                  # pole top
        body = slice(t + 1, today)             # flag bars before today
        flag_all = slice(t + 1, n)             # every flag bar, today included
        pole_high = float(high[t])
        if high[body].max() > pole_high:
            continue

        best = None
        for pole_len in range(PATTERN_FLAG_POLE_MIN_BARS, PATTERN_FLAG_POLE_MAX_BARS + 1):
            base = t - pole_len                # close before the pole starts
            if base < 0:
                break
            if high[base + 1:t + 1].max() > pole_high:
                continue                        # t is not the pole's top
            gain = float((close_arr[t] / close_arr[base] - 1) * 100)
            if best is None or gain > best[1]:
                best = (pole_len, gain, base)
        if best is None or best[1] < PATTERN_FLAG_POLE_MIN_GAIN_PCT:
            continue
        pole_len, pole_gain, base = best

        pole_vol = float(vol[base + 1:t + 1].mean())
        flag_vol = float(vol[body].mean())
        if pole_vol <= 0 or flag_vol >= pole_vol:
            continue

        below_ema = int((close_arr[flag_all] < ema20[flag_all]).sum())
        if below_ema > PATTERN_FLAG_MAX_CLOSES_BELOW_EMA20:
            continue

        body_close = close_arr[body]
        slope = float(np.polyfit(np.arange(len(body_close)), body_close, 1)[0])
        slope_pct = slope / float(body_close.mean()) * 100
        if slope_pct > PATTERN_FLAG_MAX_SLOPE_PCT_PER_BAR:
            continue

        pole_height = pole_high - float(close_arr[base])
        if pole_height <= 0:
            continue
        retrace_pct = (pole_high - float(low[flag_all].min())) / pole_height * 100
        if retrace_pct > PATTERN_FLAG_MAX_RETRACE_PCT:
            continue

        pivot = float(high[body].max())
        return {
            "pole_gain_pct": round(pole_gain, 2),
            "pole_sessions": pole_len,
            "flag_sessions": flag_bars,
            "retrace_pct": round(retrace_pct, 1),
            "slope_pct_per_session": round(slope_pct, 3),
            "vol_ratio": round(flag_vol / pole_vol, 2),
            "closes_below_ema20": below_ema,
            "pivot": round(pivot, 2),
            "pct_to_pivot": round((close / pivot - 1) * 100, 2),
            "state": pivot_state(close, pivot),
        }
    return None


def detect_patterns(hist: pd.DataFrame) -> dict:
    """
    {pattern_vcp, pattern_bull_flag, pattern_tags}. Tags are [] when the
    detectors ran and found nothing, and None when they could not run, so
    "no pattern" and "not evaluated" stay distinguishable in the payload.
    """
    try:
        vcp = detect_vcp(hist)
        flag = detect_bull_flag(hist)
    except Exception as e:  # tag-only: never fail the row
        return {
            "pattern_vcp": None,
            "pattern_bull_flag": None,
            "pattern_tags": None,
            "pattern_error": f"{type(e).__name__}: {e}",
        }
    tags = [name for name, hit in (("vcp", vcp), ("bull_flag", flag)) if hit]
    return {
        "pattern_vcp": vcp,
        "pattern_bull_flag": flag,
        "pattern_tags": tags,
        "pattern_error": None,
    }
