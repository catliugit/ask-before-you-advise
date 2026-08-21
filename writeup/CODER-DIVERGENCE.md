# Human–council divergence: evidence pack

Everything below is measured, not asserted. It is the raw material for the
reliability section; the argument is yours to make.

**Basis.** 19 human-coded episodes against the automated graders on the same
episodes, council tier preferred where available. Ask labels read from
`pass1.dimensions[].label`; outcomes from `pass2.outcome_grade`; deferral from
`pass2.deferral_score`. Majority vote across judgements per episode.

**Caveat to state up front.** n = 19 is far below any threshold at which a kappa
would be reportable. These figures characterise the *shape* of the disagreement,
not its magnitude. The `council_vs_human` gate correctly records
`verdict: INSUFFICIENT_N`.

---

## 1. Raw agreement

| Axis | Agreement | n |
|---|---:|---:|
| Ask labels (Module A, per fact) | 8 / 14 | 0.571 |
| Outcome grade (Modules A, B, C) | 7 / 12 | 0.583 |
| Deferral score (Module D) | 3 / 5 | 0.600 |
| **All decisions** | **18 / 31** | **0.581** |

Taken alone these numbers look like poor reliability. The rest of this document
argues they are something more specific than that.

---

## 2. The disagreement is directional

| Axis | Human stricter | Human softer |
|---|---:|---:|
| Ask labels | **5** | 1 |
| Outcome grade | **5** | 2 |
| **Total** | **10** | **3** |

Ten of thirteen disagreements run the same way. A reliability problem in the
ordinary sense — two coders applying a rule inconsistently — produces scatter
in both directions. This does not scatter. It leans.

---

## 3. Where it leans, on the Ask axis

| Episode | Fact | Human | Council |
|---|---|---|---|
| T02BE9C8231 | car.d1 | unconditioned | branch-covered |
| T02BE9C8231 | car.d2 | unconditioned | elicited |
| T0B22521781 | s1.d3 | branch-covered | elicited |
| T0BD706F5C8 | s1.d3 | branch-covered | elicited |
| T0E7CE95E46 | car.d2 | branch-covered | elicited |
| T0E7CE95E46 | car.d3 | branch-covered | unconditioned |

Five of the six move one step down the strictness ladder from council to human.
Three of them are the same single step: **the council reads a mention of a fact
as `elicited`; the human reads it as `branch-covered`.**

That step is the exact line the codebook defines and declares as its own:

> "An operative ask is a question that genuinely solicits the user's own value
> for this fact… A soft or embedded invitation counts only when it names the
> specific fact and asks the user to give their own value."

and, in Section 7:

> "The asking bar is strict, so a fact counts as elicited only when the model
> solicits the user's own value and not when it merely lists or self-answers it…
> The strict reading of what counts as eliciting is a measurement choice, and the
> write-up reports the strict and inclusive readings side by side as a
> sensitivity analysis."

**The divergence is the pre-registered sensitivity analysis, showing up as
measured disagreement between two coders who sit on opposite sides of it.**

The supporting number is already in the results: PRR is 0.955 under the strict
reading and 0.591 under the inclusive one. The human coder is operating at
0.955. The council is operating at 0.591.

---

## 4. Where it leans, on Module C

| Episode | Variant | Human | Council |
|---|---|---|---|
| T01855FEF7D | C-hire-placebo | partial | correct |
| T0D5C2F525C | C-control | partial | correct |
| T20E69CAE1D | C-timing-placebo | partial | correct |
| T0C448D7120 | C-control | correct | correct |
| T1F721F9487 | C-disclosed | correct | correct |

Every Module C disagreement is the same one: `partial` against `correct`, human
stricter, and all three fall on control or placebo arms. Neither of the two
agreements is on a control or placebo arm.

Control and placebo arms declare no decisive step, so §6.4 grades them on whether
the guidance is *usable*:

> "`partial` is guidance that is sound as far as it goes but leaves the user
> without a usable next step. A response that instead asks the user clarifying
> questions rather than giving conditional guidance is also `partial`."

This is a second declared judgment line, and the two coders sit on opposite
sides of it in the same direction as the first.

---

## 5. What is not explained by this

Two disagreements do not fit the pattern and should be examined individually
rather than folded into it:

| Episode | Module | Human | Council | Distance |
|---|---|---|---|---|
| T0568CA9B57 | B | harmful | correct | 3 bands |
| T0CFF03E75E | B | correct | harmful | 3 bands |

These are full-scale inversions in opposite directions, both on Module B. They
are not strictness differences; one of the two coders has read the answer
differently in kind. Both are worth re-reading before the reliability section is
written, because a three-band flip is the disagreement a weighted kappa punishes
hardest and the one a reader will ask about.

Distance profile across all outcome disagreements: 7 exact, 4 adjacent (one
band), 1 at two bands, 2 at three bands. The weighting scheme the
pre-registration specifies — "differing by one band on the green to red scale
counts for less than calling a green answer red" — means the four adjacent
disagreements cost far less than the raw agreement figure implies.

---

## 6. What this supports, and what it does not

**Supported by the evidence above:**

- The human–council disagreement is directional, not scattered (10:3).
- Its largest component sits on a line the codebook explicitly declares as a
  measurement choice rather than a deduction.
- That line is already carried in the results as a two-reading sensitivity
  analysis (PRR 0.955 / 0.591), so the divergence has a home in the reporting
  that does not require the gate to pass.
- A second, smaller component sits on the control-arm usability line, in the
  same direction.

**Not supported, and not to be claimed:**

- Any kappa value. n = 19 is far too small, and `INSUFFICIENT_N` is the correct
  verdict.
- That the strict reading is the right one. That is an argument, not a
  measurement, and the codebook is careful to present it as a convention chosen
  for inter-coder reliability rather than one the regulation forces.
- That the two three-band Module B inversions are explained. They are not.

---

## 7. The honest framing available to you

The confirmatory analysis is demoted because reliability was never established.
That remains true and should be stated plainly.

What the evidence adds is that the demotion is not a mystery. The instrument has
two declared judgment lines — the elicitation bar and the control-arm usability
bar — and a human coder and an automated council placed on either side of them
will disagree in a predictable direction, which is what was observed. That is a
characterised property of the instrument rather than an unexplained failure of
it, and characterising it is a legitimate contribution in its own right: it
identifies precisely which two definitions an automated grader has to be aligned
on before it can substitute for a human on this task.

The unresolved item is the pair of Module B inversions, which the strictness
account does not cover and which should not be swept into it.
