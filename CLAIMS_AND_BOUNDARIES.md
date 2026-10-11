> **English** ｜ [中文](CLAIMS_AND_BOUNDARIES_cn.md)

# Claims and Boundaries

> **Stance stated up front (this project's philosophical judgment)**: **A physical system is still
> fundamentally a signal-processing system and is still bound by information theory;
> the information carried by a physical field cannot be transmitted in any way that violates
> information theory, which thereby indirectly constrains the evolution of that physical field.**
> It is a **guiding judgment** (**not** a proven theorem ✓) — what the table below lists are its
> **testable corollaries** ✓
> (for the full development see `README.md` §1.1 and `md/information mechanics derivations.md` ✓)

> **The whole picture on one page**: what this project **can do, cannot do, where the evidence is,
> and how wide the scope is**.
> Updated 2026-10-02. **Before citing any number**, read this table and the conclusion-credibility
> index in `CREDIBILITY.md` §3 ✓
>
> **Positioning**: this is a **reproducible research/engineering demonstration prototype** (one
> whole chain runs end to end, the numbers are auditable, the boundaries are on the record),
> **not** a product-grade system, and **not** "using AI to discover new physics" ✗

---

## I. What it can do (backed by gates and measurements)

| Capability | Measured | Evidence / reproduction command |
|---|---|---|
| **Requirement text → q structure** (attribute set / direction / abstention) | In-domain LOOCV **87.0%**; strict LODO inductive **82.8%**; direction **98.5%**; **abstention 13–17% routed to humans** | `python b_network/bv1_train_eval.py`, `python b_network/v31_lodo_check.py` |
| **q → parameter inversion** | **21 families** (three-tier priority; the 5 non-monotonic families are **honestly not force-solved**) | `python pipeline/inverse_fisher.py` |
| **Requirement → design distribution (C)** | **4/4 design domains** (20 seeds each, dual gate): landau 88.4% / rayleigh 96.8% / chain 82.8% / array 76.2% | `python pipeline/v3*_c_check.py` |
| **Design → verification (three-state)** | **27/27 families** registered; **anything outside the domain is always `unknown`**; **the 27 families' measurements agree with ground truth (≤10% ✓)** | `python sim/verify_api.py`, `python sim/registry_truth_scan.py` |
| **Verification carries "certificate + uncertainty"** | ✅ Every `pass`/`fail` comes with its own: **uncertainty u**, **distance to the decision boundary (in units of u)**, **robustness** (robust／marginal), **certificate** (model, source of tolerance, validity domain, support measure, **known bias**, **whether third-party external corroboration already exists**, **decision rule**, **reproducible design inputs**) ⟹ **you can judge whether that `pass` can be trusted without reading the code** ✓; `unknown` also carries a certificate (stating "why unknown") ✓ | `python sim/certificate_check.py`, `python tools/demo_minimal.py` |
| **Verification feedback loop** | reflow → DB5 (390 records) + weighted dataset → consumed by training | `python pipeline/demo_end_to_end.py --pair DB3-0009` |
| **Full-chain gate** | fast tier **38 stages = PASS 36 / boundary 2 / known-issue 0 / FAIL 0**; suite **23/23**; core **17/17** | `python run_system.py` |
| **Third-party external corroboration** | **5/27 families** ✓: `diffusion/q7` (three external methods, κ̂ difference **0.00%** ✓), `two_level`, `rayleigh`, `weibull`, `landau` (**closed form／numerical differentiation-integration／Monte Carlo fraction-squared** three routes mutually consistent ✓, project vs external **0.00%** ✓) | `python sim/external_check_diffusion.py`, `python sim/external_check_five_families.py` |

---

## II. What it cannot do (and why)

| Not to be used for | Reason (**in one sentence**) |
|---|---|
| **Publicly claiming that "a cross-domain unified framework is established"** | **Analytic side**: strict LODO ✓; **representation side**: absolute capacity is not predictable ✗, but the **theoretical invariant (`log₁₀ β@90%`) passes cross-family certification at 3.0%** ✓ ⟹ the statement must be made in **layers**, and one cannot say flatly that "it is established" ✓ |
| **Taking `PASS` directly as the basis for a design decision** | The verifier is **self-built** ("outside the design loop" ≠ "third-party independent"), and third-party corroboration covers only **5/27 families** ✗ |
| **Embedding in a production system** | No service-oriented API; manual review of anchors **has not reached 100%**; **real requirement text has not been evaluated** ✗ |
| **Any claim of RL capability** | RL is only an **interface and a gate**, with **no performance claim** (the measured SAC/v14 gains do not hold) ✗ |
| **Fully automatic text → q** | Slot prediction is already automatic ✓, but **numeric extraction/rule matching still has semi-automatic steps** ✗ |
| **Citing as experimental fact** | The data is **AI-generated + manually spot-checked**, **not authoritative experimental data** ✗ |

---

## III. Known **negative results and boundaries** (not concealed)

| Item | Conclusion |
|---|---|
| **T2 family-level capacity extrapolation (absolute quantity)** | ❌ **Not passed** (currently measured median **64.0%**, criterion ≤20%). ✗ The logarithmic scale does not work ✗; profile features bring the family level down to **17.8%** but the interval crosses the threshold ⟹ not certified ✗; adding row-level samples **cannot** tighten the interval (the CI is determined by the **number of families** ✗) |
| **✅ T-c theoretical-invariant cross-family prediction** | **Certified and passing** ✓: the family-level leave-one-out median error for predicting **`log₁₀(β@90%)` (the saturation scale of the IB curve)** is **3.0%**, bootstrap 95% interval **[1.4%, 6.6%]** ⟹ **"cross-domain unification" holds in the correct coordinates of information geometry** ✓ (**caveat**: the features include the input-side `I_XT` saturation scale ⟹ this is a claim of **input↔output scale consistency**, not "prediction from scratch" ✓). Two other targets: efficiency ratio 22.5% (interval crosses the threshold ✗), knee sharpness 8.5% (**does not beat the null model** ✗) |
| **⚠️ Data version (citation warning)** | **T2 as measured today: 64.0% median** (numbers measured before the DB2 curve fill are **not reproducible**) ✗ ⟹ **whenever a number is quoted, its data version must be stated** (β 109 ✓, M4 PASS ✓, T2 as above ✓) |
| **T1 family-level fingerprint transfer** | ✅ Passed (but T2 does not pass ⟹ overall it only counts as partially established) |
| **M6.2 structural editing** | ✅ **Boundary stable** (4.50 / 1.75 / 2.00 / 2.00, reproducibly) |
| **SAC single-peak search** | ❌ **73.6% ± 32.7, bistable ⟹ not usable** |
| **v14 structural-space capability curve** | ❌ **40–45% pass rate ⟹ not usable** |
| **Feedback edge** | ❌ **0 positive / 1 harmful / 7 undetermined ⟹ not viable** |
| **Variational layer L2 increment** | ❌ **No practical increment** on either target quantity |
| **RL baseline** | ES mean **worse than random** (R3) |
| **diffusion grid alignment bias** | ⚠️ Known **0.512%** (probes sit on nodes rather than exact L) — not a failure, but **must be reported together with the conclusion** |
| **Registry measurement vs node truth (all registered families)** | ✅ **All 27 registered families: measurement agrees with node truth within 10%** — per-family deviation **0.00%–1.32%** ✓; registry calibration **27/27** ✓ (`gaussian_mixture2`／`landau`／`truncated_gaussian`: **0.20%／0.61%／0.00%** ✓) — **standing gate** |

---

## IV. Scope statement (must be included whenever citing)

1. **Scope of the cross-domain claim**: **only the "analytic/text side"** (strict LODO inductive 82.8%); **the representation/capacity side does not hold** ✓
2. **Scope of verification**: **5/27 families** have third-party corroboration; **the remaining 22 families are "outside the design loop · self-built"** ✓
3. **Scope of the data**: DB1–DB5 are **AI-generated + manually spot-checked/reviewed**; DB1 machine gate **300/300**, human spot-check **list 60/300 (= 20%)**, **review progress pending 294 / spot_checked 6**, DB3 validation **169/169**,
   DB2 usable curves **99.7%** ✓ **none of them is authoritative experimental data** ✓
4. **Scope of RL**: **provides an interface only**, **no performance claim** ✓
5. **Scope of reproduction**: single-machine Python; **no external service dependencies**; fast tier ~6–7 minutes ✓

---

## V. What is still missing to advance to "usable for internal design decisions" (ordered by cost-effectiveness)

| No. | Item | What it buys |
|---|---|---|
| **1** | **"Certificate + uncertainty" in the verification output** (model equations/discretization/source of tolerance/validity domain/known bias) | ✅ **In place**: `PASS` carries `u`, distance to the boundary, robustness and a certificate, so it reads as **evidence** rather than a conclusion ✓ |
| **2** | **Advance DB1 manual review to 100%** (currently **294 pending / 6 spot_checked**) | Anchor credibility goes from 20% sampling to full coverage ✓ |
| **3** | **Third-party external corroboration** (rayleigh / landau / two_level / weibull …) | ✅ **In place for 5/27 families** ✓; the remaining 22 families stay **self-built, outside the design loop** ✗ |
| **4** | **Real requirement-text evaluation set** (20–50 public specification sentences) | Answers "can it be used on real text" ✓ |
| **5** | (engineering, beyond the demo scope) service-oriented API, RL performance, fully automatic text→q | — |

---

## VI. Summary in one sentence

> **Can**: run the whole chain 「requirement → q → parameters → design → verification → feedback」 end to end, with auditable numbers and the boundaries on the record ✓
> **Cannot**: treat its `PASS` as external endorsement, treat RL as a capability, or treat the data as experimental fact ✗
> **The key step for an upgrade**: **verification certificate + uncertainty** (so that conclusions can be trusted by outsiders) ✓
