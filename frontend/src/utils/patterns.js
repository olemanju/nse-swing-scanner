// 1.9.0 chart-pattern tags (VCP / bull flag), pure helpers.
//
// Tag-only: these never feed the gates or the score, and the tracker has not
// measured them yet. Copy here must not imply an edge (see AGENTS.md, "There
// is no established edge for any band"). The scanner publishes:
//   pattern_tags      [] = evaluated, nothing found; null = not evaluated
//   pattern_vcp       detail object or null
//   pattern_bull_flag detail object or null

export const PATTERN_LABELS = {
  vcp: "VCP",
  bull_flag: "Bull flag",
};

const STATE_LABELS = {
  forming: "forming",
  at_pivot: "at pivot",
  breakout: "breakout",
};

const STATE_TONES = {
  forming: "muted",
  at_pivot: "accent",
  breakout: "success",
};

export const PATTERN_CAVEAT =
  "Tag only: not part of the gates or the score, and no measured edge yet.";

const num = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v, d = 1) => (num(v) ? `${v.toFixed(d)}%` : "—");
const signedPct = (v, d = 1) =>
  num(v) ? `${v > 0 ? "+" : ""}${v.toFixed(d)}%` : "—";
const inr = (v) =>
  num(v) ? `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : "—";

export function stockPatternTags(stock) {
  const tags = stock?.pattern_tags;
  if (!Array.isArray(tags)) return [];
  return tags.filter((t) => Object.prototype.hasOwnProperty.call(PATTERN_LABELS, t));
}

export function hasPattern(stock) {
  return stockPatternTags(stock).length > 0;
}

export function patternDetail(stock, tag) {
  if (!stock) return null;
  const d = tag === "vcp" ? stock.pattern_vcp : tag === "bull_flag" ? stock.pattern_bull_flag : null;
  return d && typeof d === "object" ? d : null;
}

function summary(tag, d) {
  if (tag === "vcp") {
    const depths = Array.isArray(d.depths_pct) ? d.depths_pct.map((x) => `${x}%`).join(" → ") : "";
    return [
      num(d.contractions) ? `${d.contractions} contractions` : "",
      depths,
      num(d.vol_dryup_ratio) ? `volume ${d.vol_dryup_ratio.toFixed(2)}× first leg` : "",
    ].filter(Boolean).join(" · ");
  }
  return [
    num(d.pole_gain_pct) ? `pole +${d.pole_gain_pct.toFixed(1)}% in ${d.pole_sessions} sessions` : "",
    num(d.flag_sessions) ? `flag ${d.flag_sessions} sessions` : "",
    num(d.retrace_pct) ? `retraced ${d.retrace_pct.toFixed(0)}%` : "",
  ].filter(Boolean).join(" · ");
}

/** Chips for the card / table: [{ key, label, tone, tooltip }]. */
export function patternChips(stock) {
  return stockPatternTags(stock).map((tag) => {
    const d = patternDetail(stock, tag) || {};
    const state = Object.prototype.hasOwnProperty.call(STATE_LABELS, d.state) ? d.state : null;
    const pivotTxt = num(d.pivot)
      ? `Pivot ${inr(d.pivot)} (${signedPct(d.pct_to_pivot)} away)`
      : "";
    const tooltip = [summary(tag, d), pivotTxt, PATTERN_CAVEAT].filter(Boolean).join(". ");
    return {
      key: tag,
      label: state ? `${PATTERN_LABELS[tag]} · ${STATE_LABELS[state]}` : PATTERN_LABELS[tag],
      tone: state ? STATE_TONES[state] : "muted",
      tooltip,
    };
  });
}

/** Label/value rows for the detail drawer, one list per detected pattern. */
export function patternRows(stock, tag) {
  const d = patternDetail(stock, tag);
  if (!d) return [];
  const common = [
    ["State", STATE_LABELS[d.state] || "—"],
    ["Pivot", inr(d.pivot)],
    ["Close vs pivot", signedPct(d.pct_to_pivot, 2)],
  ];
  if (tag === "vcp") {
    return [
      ...common,
      ["Contractions", num(d.contractions) ? String(d.contractions) : "—"],
      ["Pullback depths", Array.isArray(d.depths_pct) ? d.depths_pct.map((x) => `${x}%`).join(" → ") : "—"],
      ["Volume, last vs first leg", num(d.vol_dryup_ratio) ? `${d.vol_dryup_ratio.toFixed(2)}×` : "—"],
      ["Pivot below 60-session high", pct(d.pivot_below_high_pct)],
      ["Base length", num(d.base_sessions) ? `${d.base_sessions} sessions` : "—"],
    ];
  }
  return [
    ...common,
    ["Pole", num(d.pole_gain_pct) ? `+${d.pole_gain_pct.toFixed(1)}% in ${d.pole_sessions} sessions` : "—"],
    ["Flag length", num(d.flag_sessions) ? `${d.flag_sessions} sessions` : "—"],
    ["Retracement of pole", pct(d.retrace_pct, 0)],
    ["Flag slope", num(d.slope_pct_per_session) ? `${signedPct(d.slope_pct_per_session, 2)} / session` : "—"],
    ["Volume, flag vs pole", num(d.vol_ratio) ? `${d.vol_ratio.toFixed(2)}×` : "—"],
    ["Closes below 20 EMA", num(d.closes_below_ema20) ? String(d.closes_below_ema20) : "—"],
  ];
}
