> **English** ｜ 中文 (full internal plan, not shipped)

# Cross-Domain Unified Design System: Architecture, Database Design, and Implementation Plan

## 1. Purpose and scope

This document is the **application-layer engineering plan** for information mechanics theory. It does not try to
verify the theory; it uses the chain **DPI → Fisher metric → information Lagrangian** as a **hard constraint
structure** for an AI system, and builds a cross-domain unified design system on top of it.

**Positioning.** A **reproducible research/engineering demonstration prototype** — one chain runs end to end, the
numbers are auditable, the boundaries are on the record. **Not** a product-grade system, and **not** "using AI to
discover new physics": the accurate self-description is **structured search under strict information-geometry
constraints**. **RL is an optional layer and carries no performance claim**, reserved for *non-informational*
objectives only; research code sits in `a_network/`, **not on the main chain**.

## 2. Current architecture

### 2.1 The main chain, in seven steps

```
① requirement text
② → B parser  (abstention routed to a human queue)
③ → reward-type classification  (information-quantity / convention / constraint)
④ → design  (state = design vector; design-space registry)
⑤ → C generator  (requirement → design distribution)
⑥ → three-state verification with uncertainty + certificate
⑦ → feedback loop  (DB5 + weighted dataset)
```

**① Requirement text.** A domain requirement in natural language (materials, plasma, acoustics, …).

**② B parser — the single semantic entry point.** `B-parse` (neural parser; a small 4-layer encoder for slot
filling) emits a `StructuredRequirement`: slot JSON plus `abstain` plus
`ambiguity_flags`. It handles **language variability only**; decisions stay deterministic in a frozen rule engine
(`B-map`). Measured: in-domain LOOCV **87.0%**, strict LODO inductive **82.8%**, direction **98.5%**,
**abstention 13–17% routed to humans**. **Abstention is a designed outcome, not a failure**: on a rule-base miss B
returns `UNMAPPED` with a nearest-neighbour suggestion, **hard mapping is forbidden**, and the requirement goes to
the **human queue**, whose resolution is written back to DB3.

**③ Reward-type classification, promoted to a first-class citizen** (`pipeline/reward_types.py`).
"Reward = β·I_TY − I_XT is universal" is wrong: β is a Lagrange multiplier and only means something for one
objective family. Over the 159 DB3 anchors:

| Class | Share | Requirement shape | Correct mechanism | β |
| --- | --- | --- | --- | --- |
| **information-quantity** (`param`) | **73 / 159 = 46%** | "relative uncertainty ≤1%", "resolution" | IB-type `β·I_TY − I_XT` | ✅ applies (β ∈ 0.75–5.61, median 1.51) |
| **rate-type** (`q9`/`q7`) | **52 / 159 = 33%** | "95% within 20 ms" | conversion via a **documented reference scale** | ⚠️ **convention, not theorem** |
| **constraint** (`barrier`/`env`) | **34 / 159 = 21%** | "free-energy barrier ≥40 k_B T" | **satisficing** `R=1 if q≥q_req` | ❌ does not apply |
| unregistered | — | — | `unknown` — no β, no pretending to satisfice | ❌ |

The criterion has a **single source of truth** (`PROP_TO_CLASS`), which `beta_calibration._info_class`
**delegates** to, so there is no two-source drift; it agrees with `a_network/req_env.py`'s
`CONSTRAINT_REQUIREMENTS`, which runs satisficing. Forcing constraints into β yields **absurd values** (β≈0 or
even β<0 — negative β rewards *taking less information*): the **12** apparent "unlocks" of this kind are
`invalid`/`degenerate`, and `β ≤ 0`, `|β| < 1e-3` or `β > 1e3` is marked the same way rather than treated as a
successful calibration; and conversions must respect dimensions and reference scale (DB3's `q1` is often a
**dimensional directional Fisher** and must be divided by the pair's own `reference_scale`, while a requirement's
`q1 = 1/(nσ²)` already contains the sample count and needs reducing to a **single observation**) — otherwise you
get false-large numbers such as 25 nats.

**④ Design — state = design vector/field** (`pipeline/design_space.py`, `DOMAINS` as the single source of truth).
The design is continuous and high-dimensional. **Five design domains are registered, two of them mixed**
(discrete + continuous):

| Domain | dof | Requirement types | Solution structure | Mixed? |
| --- | --- | --- | --- | --- |
| `two_level` | `(ΔF, T)` | information-quantity + constraint | **1-D ray** (200 designs/state) | — |
| `rayleigh` | `(D, f)` | information-quantity | **1-D ray** | — |
| `landau` | `(a, b, kT)` | information-quantity + constraint | 3 dof → scalars `(q1, ΔF)` | — |
| `chain` | 4–6 dof (components + params) | information-quantity | chain; σ must increase along it (DPI-legal, legal rate **41%**) | **yes** |
| `array` | `N, d/λ, taper, f` (4 dof) | information-quantity | `σ_eff` cross-checked against `rayleigh` at **0.00%** | **yes** |

It provides `design_vector`/`from_vector`, `verify_design` and `feasible_set` (the solution manifold).
`two_level` and `rayleigh` are **non-degenerate**: one information state maps to a one-dimensional solution
manifold (200 distinct designs), because the information quantity fixes only a *ratio* while the absolute
requirement fixes the *scale*.

**⑤ C generator — requirement → design distribution.** C emits a **requirement → design distribution**; A acts as
a **conditioning encoder**, feeding condition vectors to C. Measured with 20 seeds and dual gates: `landau`
**88.4% ± 3.0%** (20/20, no in/out-of-interpolation gap) · `rayleigh` **96.8% ± 1.3%** (20/20) · `chain` **82.8%**
(20/20, plus diversity 160) · `array` **76.2% ± 1.9%**. MDN coverage is **96.6%** against CFM **95.4%** on the
same basis: both are far above the backbone switch-condition, so the form is **not** switched on the strength of
one domain.

**⑥ Three-state verification with uncertainty and certificate** (`sim/verify_api.py`):
`verify(design) -> Verdict`, with `verdict ∈ {pass, fail, unknown}`.

- **`unknown` is a first-class citizen**: unregistered family, unsupported metric, parameters outside the validity
  domain, missing target, adapter exception, non-finite value ⟹ **always `unknown`, never `pass`**. *"Not
  verified" ≠ "passed"* — the whole difference between demo-grade and production-grade.
- **Tolerance is tied to estimator noise**: `tol = max(floor, k/√n)`, where `k` is calibrated on **known ground
  truth** (calibration uses truth; verification does not). Runs are reproducible (design hash + seed), cached, and
  logged to an append-only audit `sim/verify_audit.jsonl`.
- **Registry** (`sim/registry.json`): registration requires p99 relative error **≤10%**, otherwise a family
  **stays unregistered and verification returns `unknown`**. Coverage is **27/27 families registered**, per-family
  deviation from node truth **0.00%–1.32%**, out-of-domain always `unknown`.
- **Certificate**: every `pass`/`fail` carries `u`, `margin_u` (distance to the decision boundary in units of `u`),
  `robustness` (`|margin_u| ≥ 1` ⟹ `robust`, else `marginal`), and a `certificate` (family, metric, model, method,
  tolerance and its source, validity domain, supported metrics, registry source, **known bias**, **external
  corroboration**, verdict rule, reproducible design). `margin_u > 0 ⟺ pass`, via the *same* formula, so no new
  convention is introduced; `unknown` also gets a certificate, stating **why** it is unknown.
- **External corroboration stands at 5/27 families** (project vs. external **0.00%**). A self-built verifier is
  "outside the design loop", not a third party — `PASS` must not be read as external endorsement.

**⑦ Feedback loop** (`pipeline/design_reflow.py`):

```
C generates design distribution → sampling → external verification → filter/weight → feedback (persist / retrain)
```

**Entries that did not pass or were not verified do not enter the feedback set** (the production-grade rule
`unknown ≠ pass`), so backflow is **evidence-based retraining**, not "feeding noise back into training".
Output is an append-only `design_reflow_<domain>.jsonl` (pass set + weights + statistics, auditable); it runs in
two domains — solution-manifold sampling for `rayleigh`/`two_level`, design-box sampling for `landau` — and is
consumed downstream as **DB5 + a weighted dataset (DB5: 390 records)**. Tolerance conservatism: p99 ground-truth
error **1.84%** against a tolerance of **8.5%** ⟹ **≈4.6× conservative** (safe against false `pass`, weak in
verification power); with the tolerance recalibrated to **3.6%**, conservatism is **2.0×**.

### 2.2 The layers

① axioms — DPI → Fisher → information Lagrangian; ② data/parse — DB1–DB5 → B parse → q vector; ③ calibration —
reward-type classification → β / satisficing / convention; ④ design — design-space registry, state = design
vector; ⑤ generation — C: requirement → design distribution; ⑥ verification — production API, 27 families,
out-of-domain = `unknown`, audited; ⑦ feedback — verification feedback loop (filter/weight → retrain); ⑧ optional
optimisation — RL, **non-informational objectives only**. Of the pre-registered criteria, R1 (`landau`
88.4% ± 3.0%, 20/20), R2 (2 non-degenerate domains), R3, R6, R7 and R15 (27/27 families) are met; R4 has the
module present with the run pending; R5 (1 → 3–5 domains) stands at 3.

## 3. Premises that do not hold

These four premises are the reason the main chain is C-dominant; each is a measured result, not a preference.

| Premise | What is measured | Nature |
| --- | --- | --- |
| "RL searching the information-structure space ⟹ better designs" | **Contradicts DPI** — the longer the chain, the less information; 3-node `I_TY` median **1.15e-07** | **Theorem-level; cannot be engineered around** |
| "A-network embeddings can serve as the RL state" | That is a **representation**, not a **design**; real designs are continuous and high-dimensional — `(ΔF,T)`, `(D,f)`, `V(φ)` | Level mismatch (**fixable**) |
| "Reward = β·I_TY − I_XT is universal" | **β only applies to the information-quantity family** (**73/159 = 46%**); 33% is convention, 21% must be satisficing | Wrong objective-family choice (**fixable**) |
| "RL → DB5 → backflow → RL is a closed loop" | Backflow edge: **0 positive / 1 harmful / 7 undetermined** ⟹ the loop is **not closed** | Broken loop (**fixable**) |

Supporting measurement for the first row: the evaluable rate by node count is **100% / 38% / 12% / 10%** for 1–4
nodes; the best 2-node chain reward is **+0.0138** against **+2.656** for a single node, about a factor of 200, so
under an IB-type reward the optimal policy is to **stop at one node** and combinatorial editing has no reward
signal (only **0 of 7** threshold tests are satisfied). By DPI each edge can only add noise, accumulated variance
grows monotonically with chain length, and `MI = ½ln(1+σ_src²/Σvar_n)` falls monotonically — **"more nodes carry
more information" is not a hard problem but a contradiction of the axiom.**

Consequences: the full structural space (gap ③) is **not worth building** — combinatorial structure is valueless
under information-quantity objectives; the M6.2 "structure editing 0/20" boundary is *not* a representation
shortfall but *an objective that contradicts DPI*; and combinatorics can only pay off under a
**non-informational** objective. The representation and environment layers are **retained** for reuse under those
objectives.

## 4. The five databases

```
DB1 physical system store ──B v0 encoding──▶ DB2 fingerprint store ──training──▶ network A
manual seeds + LLM expansion ──▶ DB3 requirement-mapping store ──training──▶ network B (v1)
RL runs ──▶ DB4 structure/experience store ──▶ C decode ──▶ DB5 validation store
                                                             │ pass only
                                                             └──▶ back to DB3 / A's fine-tune set (discounted)
```

| DB | Holds | Scale | Red line |
| --- | --- | --- | --- |
| **DB1** | Physical system entries: domain/family/parameters/equations + declarative Fisher fingerprints | **300 entries** | Machine gate **300/300** ✓; human spot-check **list 60/300 (= 20%)**, review progress **294 pending / 6 spot_checked**; **AI-generated entries may not serve as anchors** |
| **DB2** | Information fingerprints: each system's IB curve, Fisher trace, capacity | **8879 rows** (usable curves **99.7%**) | Curves and fingerprints must match the ground-truth basis (gated ✓) |
| **DB3** | Requirement-mapping anchors: requirement text ↔ q structure (attributes/direction/value) | **169 anchors** | **Requires dual-role (A/B) human review**; verified 169/169 ✓ |
| **DB4** | RL experience (historical layer) | **87 records** | Historical reference only ✓ |
| **DB5** | Decode validation results (passers written only) | **390 records** | Consistent with `verdict ∈ {pass, fail}` ✓ |

**Data nature.** All five stores are **AI-generated + human spot-checked/reviewed**; **none of them is
authoritative experimental data**.

**DB1** is built by **AI template filling plus a computability gate** — every system must run end to end through
the core information computation library and emit a fingerprint, or it is not admitted, which keeps the DB1↔DB2
interface free of mismatch. **Six mandatory symbol conventions** precede it: nats and the information functionals;
distribution families and parameterisation; Itô SDEs with `σ²` as variance rate; non-dimensionalisation and
bridging constants; graph encoding with edge direction = information-flow direction `X→T→Y`; and
strictness/provenance, where **anchor-level formulas entering DB3 must cite literature**. Convention drift does not
break any single system's self-consistency — so a fingerprint computability test **cannot catch it** — but it
poisons cross-system comparability of DB2's q vectors, so the convention specification is itself versioned
(`convention_version`) and a convention change means **re-encoding all of DB2**, on the same footing as
`encoder_version`.

**DB2** is generated by DB1 system × parameter point → B v0 encoding → information-structure graph → core
information computation → fingerprint, storing `encoded_graph`, `fisher_summary`, `mi_profile`, `ib_curve`,
`dpi_margins`, `s_info`, `encoder_version` and `compute_code_hash`. `encoder_version` is **mandatory**; a B upgrade
re-encodes the whole store and the old version is archived. This is A's **only** training source — no A sample
reaches it without passing B.

**DB3** is built in three steps: **manual seeds (20–50 pairs)** derived from the theory correspondence dictionary,
each with a `metric_rationale` citing the theory document; **rule/LLM expansion to 500–1000 pairs**; and a **human
spot check at ≥20%**, with failing batches sent back. Each record carries the requirement text and its structured
form, the normalised `info_metric_vector` q, its rationale, `source` (`manual`/`llm_expanded`/`rl_validated`),
`confidence` and `reviewer`.

**DB4** is the RL replay buffer promoted to a database (`graph`, `a_scores`, `reward_components` for
IB/DPI/A/complexity, `episode_meta`, `domain_tag`/`decoded`, and a nullable `validation_result` FK → DB5). Its 87
structural experiences break down as GFN 20 / SAC 3 / random 63 / domain instance 1. **Data hygiene**: unverified
DB4 structures are **physically isolated** from DB2's known physics, checked by an automated CI test on every
deployment — the guard against self-reinforcing hallucination.

**DB5** records the validation id, its DB4 structure FK, `domain`/`decoded_design`, `validation_method`
(`analytic`/`simulation`/`experiment`), `metrics`/`pass_fail`, and `date`/`validator`. **Only `pass` entries**
trigger (i) writing the requirement↔metric mapping back to DB3 as `rl_validated`, and (ii) admitting the structure
to A's fine-tune candidate set labelled "machine-discovered" with discounted confidence. The target of this
write-back is **C's next training round**, not the RL replay buffer: no positive-gain evidence for RL-side backflow
exists.

All five stores carry the same four-level strictness labelling, and records are traceable to their original system,
encoder version and compute-code hash. Train/test splits are made **by system family, not by parameter point**, so
generalisation tests are genuinely out-of-distribution.

## 5. Engineering and credibility standards

Every conclusion in this project is measured against the standards below.

1. **Conclusion threshold.** Any RL-side conclusion (environment, training, or control comparisons) requires **≥20
   seeds and a pre-registered criterion**. Results with fewer than 20 seeds may only be cited as **leads**.
2. **Pre-registration.** Criteria, thresholds and the three-state definitions are **written into the script before
   the data is seen**, and applied mechanically afterwards. Post-hoc selection of significant items is forbidden.
3. **Per-point persistence.** Long runs **append each seed's result to a CSV** and provide a programmatic summary
   (`--report`), so runs resume after interruption and statistics do not depend on the eye.
4. **Three-state verdict.** Pass rate **≥70% → robustly established**; **≤30% → robustly not established**;
   anything in between → **unstable (seed-dependent)**. A significant/not-significant binary must not paper over
   instability.
5. **Negative results are held to the same standard as positive ones.** M6.2's 0/20 is a **stable boundary**
   (20 seeds, robustly failed), which is what establishes "v0.1 GFN cannot do combinatorial structure editing".
6. **Density/probability models must self-check `∫p̂dφ = 1`.** No ground-truth density is needed to expose
   normalisation or integration errors. Any component that "outputs a distribution" (C decoder, world model,
   variational posterior) must report that integral and its step-size convergence before being stored or shipped.
7. **"Impossible numbers" are treated as bugs**: model NLL **below** the true distribution's NLL, coverage
   **> 1**, information exceeding the DPI upper bound. Such numbers are theoretically impossible — **check the
   implementation on sight; never write them into a conclusion or a document.**

**Citation discipline.** Every number is traceable to an entry of the credibility index
([`CREDIBILITY.md`](CREDIBILITY.md)), with its seed count and status; numbers that have not been re-checked are marked
**"not re-checked"**. Three consequences of the standards: GFlowNet (diverse-solution sampling) **may serve as the
production path** (20/20 robust, std 0.028), while **SAC's "stable single-peak search" has no supporting
evidence** (bistable, roughly 1/3 of seeds converging to the wrong mode) and is a fallback only; structural
editing is **not merely low-hit-rate — even reward improvement is unstable** (40–45%); and the backflow edge is an
**optional path**.

## 6. Negative results and current measurements

These are the numbers to cite.

| Item | Status | Current measurement |
| --- | --- | --- |
| **SAC single-peak search (M7)** | ❌ **Not usable** | **73.6% ± 32.7**, 13/20 pass, **bistable** — about 1/3 of seeds converge to the wrong mode |
| **v14 capability curve** | ❌ **Not usable** | All three σ at **0.56–0.74 ± ~1.0**, pass rate **40–45%** |
| **Backflow edge** | ❌ **Not viable** | **0 positive / 1 harmful / 7 undetermined**; supervised pretraining (Q-regression warm start) is not used — harmful evidence on 20 seeds; the edge is **optional** |
| **T2 absolute-capacity cross-family transfer** | ❌ **NOT passing** | Median **64.0%** against a criterion of ≤20%. ✗ log basis invalid ✗; profile features bring the family level to **17.8%** but the interval crosses the threshold ⟹ not certified ✗; adding row-level samples **cannot** tighten the interval, because the CI is governed by **family count** ✗ |
| **T-c theoretical-invariant cross-family prediction** | ✅ **Certified** | Predicting `log₁₀(β@90%)` (the IB-curve saturation scale): family-level leave-one-family-out median error **3.0%**, bootstrap 95% CI **[1.4%, 6.6%]**. Reservation: features include the input-side `I_XT` saturation scale ⟹ an **input↔output scale-consistency** proposition, **not** prediction from zero |
| **T2 as measured today** | ⚠️ **Data version** | Median **64.0%**; numbers measured before the DB2 curve fill (41.5% → 99.7%) are **not reproducible** and must carry that data-version note |
| **diffusion/q7 grid-alignment bias** | ⚠️ **Report with the conclusion** | The probe node sits at `x = 0.009975`, **0.251%** closer than `L` ⟹ `t_half` and κ̂ are biased by **0.512%** |

Related measurements: M5 (GFlowNet dual-mode coverage) **holds robustly, 20/20**, at **0.372 / 0.361** with std
**0.028 / 0.029**; M6.2 fails robustly at **0/20**, so that negative conclusion is stable; and M6.2's M3 ΔlogR is
**+1.45 ± 0.76** (17/20 pass).

Still open: any judgement resting on **1–12 seed** single-point values rather than the ≥20-seed standard, and the
claim that structural-space capability is merely *low* rather than *unstable* — it is unstable. The
**`requirement → β` calibration remains a placeholder (β = 4/1)**, and DB3 text is still coupled to the RL task
semi-automatically.

## 7. Reading map

| Document | What it is |
| --- | --- |
| [README.md](README.md) / [README_cn.md](README_cn.md) | Project overview, capability table, and the five-database summary with current scales |
| [CLAIMS_AND_BOUNDARIES.md](CLAIMS_AND_BOUNDARIES.md) / [CLAIMS_AND_BOUNDARIES_cn.md](CLAIMS_AND_BOUNDARIES_cn.md) | **Read before citing any number**: what the project can do, cannot do, where the evidence is, and how wide the scope is |
| [CREDIBILITY.md](CREDIBILITY.md) | The **credibility index**: the measured status of every headline claim, with the protocol each number belongs to |
| [LICENSE](LICENSE) | Code **MIT**; data and documents **CC BY 4.0** |
| Internal development log (not shipped) | The full 214-entry record behind every number above |
