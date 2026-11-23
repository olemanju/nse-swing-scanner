import { test } from "node:test";
import assert from "node:assert/strict";

import {
  PATTERN_CAVEAT,
  hasPattern,
  patternChips,
  patternRows,
  stockPatternTags,
} from "./patterns.js";

const VCP = {
  contractions: 3,
  depths_pct: [20.8, 11, 5.5],
  final_depth_pct: 5.54,
  vol_dryup_ratio: 0.5,
  pivot: 98.49,
  pivot_below_high_pct: 2,
  pct_to_pivot: -0.3,
  state: "at_pivot",
  base_sessions: 50,
};

const FLAG = {
  pole_gain_pct: 15,
  pole_sessions: 8,
  flag_sessions: 6,
  retrace_pct: 26.5,
  slope_pct_per_session: -0.355,
  vol_ratio: 0.33,
  closes_below_ema20: 0,
  pivot: 114.57,
  pct_to_pivot: 0.6,
  state: "breakout",
};

test("null and [] tags both mean no chips", () => {
  assert.deepEqual(patternChips({ pattern_tags: null }), []);
  assert.deepEqual(patternChips({ pattern_tags: [] }), []);
  assert.deepEqual(patternChips(undefined), []);
  assert.equal(hasPattern({ pattern_tags: [] }), false);
});

test("unknown tags are ignored (forward compatibility)", () => {
  assert.deepEqual(stockPatternTags({ pattern_tags: ["cup_handle", "vcp"] }), ["vcp"]);
});

test("chips carry pattern + state and the tag-only caveat", () => {
  const chips = patternChips({
    pattern_tags: ["vcp", "bull_flag"],
    pattern_vcp: VCP,
    pattern_bull_flag: FLAG,
  });
  assert.equal(chips.length, 2);
  assert.equal(chips[0].label, "VCP · at pivot");
  assert.equal(chips[0].tone, "accent");
  assert.equal(chips[1].label, "Bull flag · breakout");
  assert.equal(chips[1].tone, "success");
  for (const c of chips) assert.ok(c.tooltip.includes(PATTERN_CAVEAT));
  assert.ok(chips[0].tooltip.includes("20.8% → 11% → 5.5%"));
});

test("a tag without its detail object still renders a plain chip", () => {
  const [chip] = patternChips({ pattern_tags: ["vcp"], pattern_vcp: null });
  assert.equal(chip.label, "VCP");
  assert.equal(chip.tone, "muted");
});

test("caveat states tag-only and no measured edge", () => {
  assert.match(PATTERN_CAVEAT, /not part of the gates or the score/);
  assert.match(PATTERN_CAVEAT, /no measured edge yet/);
});

test("drawer rows for each pattern", () => {
  const stock = { pattern_vcp: VCP, pattern_bull_flag: FLAG };
  const vcpRows = Object.fromEntries(patternRows(stock, "vcp"));
  assert.equal(vcpRows.State, "at pivot");
  assert.equal(vcpRows.Contractions, "3");
  assert.equal(vcpRows["Close vs pivot"], "-0.30%");
  const flagRows = Object.fromEntries(patternRows(stock, "bull_flag"));
  assert.equal(flagRows.Pole, "+15.0% in 8 sessions");
  assert.equal(flagRows["Closes below 20 EMA"], "0");
  assert.deepEqual(patternRows({}, "vcp"), []);
});
