"""Tests for the tag-only VCP / bull-flag detectors (patterns.py).

Synthetic price paths are built from anchor points so each test states the
exact geometry it exercises. Highs/lows sit 0.5% either side of the close.
"""
import numpy as np
import pandas as pd
import pytest

import patterns
from patterns import (
    detect_bull_flag,
    detect_patterns,
    detect_vcp,
    pivot_state,
    swing_high_indices,
)


def _path(anchors, n):
    """Linear interpolation through (bar, close) anchors over n bars."""
    xs, ys = zip(*anchors)
    return np.interp(np.arange(n), xs, ys)


def _frame(close, volume):
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(close), freq="B")
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": volume,
    }, index=idx)


def _vcp_frame(last_close=97.5, depths=(80, 89, 93.5), vols=(2e6, 1.5e6, 1e6)):
    """60 bars: rally to 100, then three pullbacks to `depths`, ending at last_close.

    Swing highs at bars 10 / 30 / 44 (100, 99, 98); troughs at 20 / 37 / 49.
    """
    d1, d2, d3 = depths
    close = _path([(0, 70), (10, 100), (20, d1), (30, 99), (37, d2),
                   (44, 98), (49, d3), (59, last_close)], 60)
    volume = np.empty(60)
    volume[:30] = vols[0]
    volume[30:44] = vols[1]
    volume[44:] = vols[2]
    return _frame(close, volume)


def _flag_frame(flag_closes, flag_vol=1e6, pole_vol=3e6, pole_to=115.0):
    """60 flat bars at 100, an 8-bar pole to `pole_to`, then the flag closes."""
    flat = np.full(60, 100.0)
    pole = np.linspace(100, pole_to, 9)[1:]
    close = np.concatenate([flat, pole, np.asarray(flag_closes, dtype=float)])
    volume = np.concatenate([
        np.full(60, 1.5e6),
        np.full(8, pole_vol),
        np.full(len(flag_closes), flag_vol),
    ])
    return _frame(close, volume)


# ---------------------------------------------------------------- helpers

def test_pivot_state_bands():
    assert pivot_state(101, 100) == "breakout"
    assert pivot_state(99.6, 100) == "at_pivot"
    assert pivot_state(99.4, 100) == "forming"


def test_swing_highs_single_point_on_flat_top():
    high = np.array([1, 2, 3, 5, 5, 5, 3, 2, 1, 0, 0, 0, 0], dtype=float)
    assert swing_high_indices(high, window=2) == [3]


def test_swing_highs_ignore_unconfirmed_last_bars():
    high = np.arange(20, dtype=float)  # still rising: no confirmed top
    assert swing_high_indices(high, window=5) == []


# ---------------------------------------------------------------- VCP

def test_vcp_detected_with_three_shrinking_contractions():
    out = detect_vcp(_vcp_frame())
    assert out is not None
    assert out["contractions"] == 3
    d = out["depths_pct"]
    assert d[0] > d[1] > d[2]
    assert out["final_depth_pct"] < 8
    assert out["vol_dryup_ratio"] == pytest.approx(0.5, abs=0.01)
    assert out["pivot"] == pytest.approx(98 * 1.005, abs=0.01)
    assert out["state"] == "forming"
    assert out["pct_to_pivot"] < -0.5


def test_vcp_state_at_pivot_and_breakout():
    assert detect_vcp(_vcp_frame(last_close=98.2))["state"] == "at_pivot"
    assert detect_vcp(_vcp_frame(last_close=99.5))["state"] == "breakout"


def test_vcp_rejects_expanding_contractions():
    # Pullbacks get deeper (to 93.5, then 89, then 80): the trailing run is 1.
    close = _path([(0, 70), (10, 100), (20, 93.5), (30, 99), (37, 89),
                   (44, 98), (49, 80), (59, 90)], 60)
    assert detect_vcp(_frame(close, np.full(60, 1e6))) is None


def test_vcp_rejects_without_volume_dry_up():
    assert detect_vcp(_vcp_frame(vols=(1e6, 1e6, 0.9e6))) is None


def test_vcp_rejects_final_contraction_too_wide():
    # 98 -> 89.5 is an 8.5%+ final pullback, still shallower than the 2nd leg.
    assert detect_vcp(_vcp_frame(depths=(75, 85, 89.5))) is None


def test_vcp_needs_full_lookback():
    assert detect_vcp(_vcp_frame().tail(59)) is None


def test_vcp_uses_only_last_60_sessions():
    older = _frame(np.full(40, 50.0), np.full(40, 5e6))
    combined = pd.concat([older, _vcp_frame()])
    combined.index = pd.date_range("2025-10-01", periods=len(combined), freq="B")
    assert detect_vcp(combined) is not None


# ---------------------------------------------------------------- bull flag

GOOD_FLAG = [114, 113, 112.5, 112, 112.5, 113.4]


def test_bull_flag_detected():
    out = detect_bull_flag(_flag_frame(GOOD_FLAG))
    assert out is not None
    assert out["pole_gain_pct"] >= 8
    assert out["flag_sessions"] == 6
    assert out["retrace_pct"] <= 50
    assert out["slope_pct_per_session"] <= 0.3
    assert out["vol_ratio"] == pytest.approx(1 / 3, abs=0.01)
    assert out["pivot"] == pytest.approx(114 * 1.005, abs=0.01)
    assert out["state"] == "forming"


def test_bull_flag_breakout_on_trigger_bar():
    out = detect_bull_flag(_flag_frame(GOOD_FLAG[:-1] + [115.2]))
    assert out is not None
    assert out["state"] == "breakout"


def test_bull_flag_rejects_deep_retracement(monkeypatch):
    # Falls back to 106 against a 100 -> 115 pole: ~63% retracement.
    frame = _flag_frame([112, 109, 106, 107, 108, 109])
    assert detect_bull_flag(frame) is None
    # ...and it is the retracement rule doing the rejecting.
    monkeypatch.setattr(patterns, "PATTERN_FLAG_MAX_RETRACE_PCT", 70.0)
    assert detect_bull_flag(frame) is not None


def test_bull_flag_rejects_rising_flag():
    # Each flag close ~1% higher than the last: slope well above 0.3%/bar,
    # and the pole top stops being the high, so no flag either way.
    assert detect_bull_flag(_flag_frame([114.5, 115.6, 116.8, 118, 119.2, 120.4])) is None


def test_bull_flag_rejects_heavy_flag_volume():
    assert detect_bull_flag(_flag_frame(GOOD_FLAG, flag_vol=3.5e6)) is None


def test_bull_flag_rejects_small_pole(monkeypatch):
    frame = _flag_frame([104.5, 104.2, 104, 103.9, 104, 104.1], pole_to=105.0)
    assert detect_bull_flag(frame) is None
    monkeypatch.setattr(patterns, "PATTERN_FLAG_POLE_MIN_GAIN_PCT", 4.0)
    assert detect_bull_flag(frame) is not None


def test_bull_flag_rejects_closes_below_ema20(monkeypatch):
    # A long flag whose last three closes slip under the (risen) 20 EMA while
    # the retracement stays just inside 50%.
    frame = _flag_frame([113, 112.5, 112.5, 112, 112, 111.5, 111.5, 111, 108.6, 108.5, 108.7])
    assert detect_bull_flag(frame) is None
    monkeypatch.setattr(patterns, "PATTERN_FLAG_MAX_CLOSES_BELOW_EMA20", 3)
    out = detect_bull_flag(frame)
    assert out is not None and out["closes_below_ema20"] == 3


# ---------------------------------------------------------------- wrapper

def test_detect_patterns_tags():
    vcp_only = detect_patterns(_vcp_frame())
    assert vcp_only["pattern_tags"] == ["vcp"]
    assert vcp_only["pattern_bull_flag"] is None
    assert vcp_only["pattern_error"] is None

    flag_only = detect_patterns(_flag_frame(GOOD_FLAG))
    assert flag_only["pattern_tags"] == ["bull_flag"]


def test_detect_patterns_empty_when_nothing_found():
    flat = _frame(np.full(120, 100.0), np.full(120, 1e6))
    out = detect_patterns(flat)
    assert out["pattern_tags"] == []
    assert out["pattern_vcp"] is None and out["pattern_bull_flag"] is None


def test_detect_patterns_never_raises(monkeypatch):
    def boom(_):
        raise ValueError("bad bar")
    monkeypatch.setattr(patterns, "detect_vcp", boom)
    out = detect_patterns(_vcp_frame())
    assert out["pattern_tags"] is None
    assert "bad bar" in out["pattern_error"]


def test_detect_patterns_ignores_nan_rows():
    frame = _vcp_frame()
    frame = pd.concat([frame.iloc[:30], frame.iloc[[30]].assign(Close=np.nan), frame.iloc[30:]])
    frame.index = pd.date_range("2026-01-01", periods=len(frame), freq="B")
    assert detect_patterns(frame)["pattern_tags"] == ["vcp"]


def test_outputs_are_plain_python_numbers():
    """Payloads go through the JSON file cache; keep numpy scalars out."""
    for out in (detect_vcp(_vcp_frame()), detect_bull_flag(_flag_frame(GOOD_FLAG))):
        for v in out.values():
            items = v if isinstance(v, list) else [v]
            for x in items:
                assert type(x) in (int, float, str), (x, type(x))


# ---------------------------------------------------------------- integration

def test_compute_technicals_publishes_pattern_fields(monkeypatch):
    import technicals

    lead_in = _frame(np.linspace(55, 70, 190), np.full(190, 2e6))
    hist = pd.concat([lead_in, _vcp_frame()])
    hist.index = pd.date_range("2025-03-03", periods=len(hist), freq="B")

    class _FakeTicker:
        def history(self, period=None, auto_adjust=None):
            return hist

    monkeypatch.setattr(technicals.yf, "Ticker", lambda symbol: _FakeTicker())
    out = technicals.compute_technicals("TEST.NS")
    assert out["error"] is None
    assert out["pattern_tags"] == ["vcp"]
    assert out["pattern_vcp"]["contractions"] == 3
    assert out["pattern_bull_flag"] is None


def test_json_records_carry_pattern_fields():
    from scanner import to_json_records

    base = {"symbol": "X", "gate_pass": False}
    vcp = {"contractions": 3, "pivot": 98.49, "state": "forming"}
    evaluated = to_json_records(pd.DataFrame([{
        **base,
        "tech_pattern_tags": ["vcp"],
        "tech_pattern_vcp": vcp,
        "tech_pattern_bull_flag": None,
    }]))[0]
    assert evaluated["pattern_tags"] == ["vcp"]
    assert evaluated["pattern_vcp"] == vcp
    assert evaluated["pattern_bull_flag"] is None

    # Technicals failed: tags are null ("not evaluated"), not [] ("none found").
    missing = to_json_records(pd.DataFrame([base]))[0]
    assert missing["pattern_tags"] is None
    assert missing["pattern_vcp"] is None


def test_snapshot_keeps_pattern_tags():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import snapshot_writer

    compact = snapshot_writer.compact_scan_payload({
        "generated_at": "2026-11-20T10:31:00+00:00",
        "stocks": [{"symbol": "X", "pattern_tags": ["bull_flag"], "pattern_bull_flag": {"pivot": 1}}],
    })
    row = compact["stocks"][0]
    assert row["pattern_tags"] == ["bull_flag"]
    assert "pattern_bull_flag" not in row  # details stay in latest_scan.json only
