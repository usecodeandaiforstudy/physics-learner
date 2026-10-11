# Information Mechanics Cross-Domain Design System

> **English** ｜ [中文](README_cn.md)

> A cross-domain pipeline that turns **requirement text** into **verifiable designs**.
> Its unifying language is not domain jargon but **information**: every requirement is
> translated into the same set of information metrics (a q-vector), so the
> "requirement → design → verification" mechanism can be reused across domains.
> Theoretically it rests on a single axiom — the **data processing inequality (DPI)**.
>
> **The underlying judgment**: **a physical system is still, essentially, a signal
> processing system, and it remains subject to information theory; the information
> carried by a physical field cannot be transmitted in a way that violates information
> theory — which indirectly constrains the evolution of that field.**
> This judgment and its testable consequences are the philosophical starting point of
> the whole project (see [§1.1](#11-the-basic-judgment-the-philosophical-starting-point)).

**What it is**: a runnable research/engineering prototype. The chain runs end to end,
the numbers are auditable, and the boundaries are written down.
**What it is not**: not a production system, and not "AI discovering new physics"
(see [`CLAIMS_AND_BOUNDARIES.md`](CLAIMS_AND_BOUNDARIES.md)).

---

## Contents

1. [Core ideas: why information is the unifying language](#1-core-ideas-why-information-is-the-unifying-language)
2. [System architecture: the current main chain](#2-system-architecture-the-current-main-chain)
3. [The five databases (DB1–DB5)](#3-the-five-databases-db1db5)
4. [What it can do today](#4-what-it-can-do-today)
5. [Directories and the role of every file](#5-directories-and-the-role-of-every-file)
6. [Quick start](#6-quick-start)
7. [Data red lines (must read)](#7-data-red-lines-must-read)
8. [License, citation, further reading](#8-license-citation-further-reading)

---

## 1. Core ideas: why information is the unifying language

### 1.1 The basic judgment (the philosophical starting point)

> **A physical system is still, essentially, a signal processing system, and it
> remains subject to information theory. The information carried by a physical field
> cannot be transmitted in a way that violates information theory — and therefore
> information theory indirectly constrains the evolution of physical fields.**

This is the project's **guiding judgment** (a stance), **not a proven theorem**.
Its value lies in its **testable consequences**, and every mechanism in this repository
is an implementation of one of them:

| Decomposition of the judgment | What it implies | Where it lives in the project |
|---|---|---|
| **① A physical system = a signal processing system** | Any physical process can be written as an **information channel**: environment/initial conditions (X) → internal state (T) → observable/function (Y) | `system_encoder` encodes a physical system as an X→T→Y structure graph; `fingerprint_computer` computes its information fingerprint |
| **② Information carried by a field cannot violate information theory** | Information **transfer** must obey **DPI**: `I(X;T2) ≤ I(X;T1)` — it can only be lost, never created | `core/fisher.py`, `core/ib.py`; the runner's **DPI/Fisher constraint check** treats it as a **hard gate** |
| **③ Hence "evolution" is indirectly constrained** | Usable **resolution has an upper bound** ⟹ requirement metrics such as accuracy / response / transport all have **information-theoretic bounds**, they are not freely tunable engineering knobs | The **q-vector** (q1 accuracy / q7 front velocity / q9 spectral gap …) + **q → parameter inversion** (21 families) + **β calibration** |
| **④ Consequence: a design that violates DPI is physically impossible** | "Impossible" is **excluded a priori**, not discovered by simulation | The verifier's **three-state** verdict: out-of-domain or unsupported ⇒ `unknown`, **never `pass`**; DPI margin (q4) is computed as a checkable quantity |

**Three immediate consequences** (which are also the project's three design principles):

1. **A unifying language**: because the underlying constraint is the same information
   inequality, differences between domains are differences in **channel type**, not in
   **constraint**. Requirements can therefore be translated into the **same q-vector**,
   and parsing / inversion / verification are **reused across domains**
   (27 families, 52 text domains, one mechanism).
2. **Verifiability**: if physics is information-constrained, then "is this design
   achievable?" is a **decidable** question — verification is not "run a simulation and
   look", it is a **three-state verdict against a bound** (`pass`/`fail`/`unknown`)
   accompanied by an **uncertainty and a certificate**.
3. **Explainable boundaries**: the constraint is **universal** (DPI always holds), but
   the **concrete capacity is family-specific** (channel type, support, noise structure
   differ). This is exactly what the measured boundary shows: what transfers across
   families is the **shape/scale of the curve** (certified at 3.0%), **not the absolute
   capacity** (not passing).

> **What it does not claim**: this project does **not** claim that "new physics was
> derived from information theory", and it does **not** present this judgment as a
> proven theorem. It is a **guiding judgment** whose worth lies in **testable
> consequences** — consequences that are implemented here, guarded by gates, and
> corroborated by external methods.
> (Extended derivations and delimitation: `md/information mechanics derivations.md`,
> `md/information mechanics delimitation.md`, `md/why these limits.md`.)

### 1.2 The single axiom: the data processing inequality (DPI)

No step of information processing can **increase** information about the target:

```
I(X; T2) ≤ I(X; T1)        (X: raw data, T: intermediate representation, Y: target)
```

This inequality has **no free parameters** and is the single starting point of the
whole system. It imposes a hard constraint: a representation can only **lose**
information, never create it — so "how much to compress / how much to keep" becomes a
**computable** trade-off.

### 1.3 From axiom to metric: the Fisher information metric

On a parameter space, the unique natural metric compatible with DPI and satisfying
invariance is the **Fisher information metric**:

```
g_ij(θ) = E[ ∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j ]
```

It turns "measurement precision" into geometry: **larger Fisher = easier to resolve the
parameter**. The project implements Fisher for **27 distribution families** (analytic,
numerical, or Monte-Carlo) and calibrates each one against truth.

### 1.4 From metric to dynamics: the information Lagrangian

Treating the Fisher metric as the kinetic term and the information-bottleneck functional
as the potential term gives the **information Lagrangian**:

```
L_info = ½ · θ̇ᵀ g(θ) θ̇  −  [ I(X;T) − β·I(T;Y) ]
         └ kinetic (Fisher) ┘   └ potential (IB functional) ┘
```

(Implemented in `core/action.py`; sign conventions in `db1/conventions.yaml`.)
Its meaning: an **optimal representation is the trajectory extremizing the action**,
rather than something found by trial and error. The project does not use it as a
training objective but as the **theory-layer bridge and consistency check** (the
variational layer and theory-constraint checks keep exercising it).

### 1.5 From dynamics to engineering: the q-vector

Engineering needs an interface that is **measurable, comparable and reproducible**, so
the project defines a set of information metrics — the **q-vector**:

| Metric | Meaning | Typical use |
|---|---|---|
| `q1` Fisher | **precision** of measurement/control (information content) | sensor precision, resolution |
| `q7` front velocity | speed of **transport/propagation** | heat conduction, diffusion |
| `q9` spectral gap | speed of **response/recovery** (2nd eigenvalue of Fokker–Planck) | control loops, actuators |
| `q4/q5/q8/q10` | DPI margin / channel capacity / stability margin / redundancy | robustness, adaptability, stability |

**The key point**: a requirement ("uncertainty ≤ 0.5%", "response ≤ 20 ms", "conduct heat
fast") becomes a bound on a few q-quantities. **Cross-domain reuse happens at this
layer** — the physics differs between domains, but the semantics of the q-quantities are
the same, so criteria, inversion and verifiers can be shared.

### 1.6 IB curves and the capacity/cost trade-off

Each system's information behaviour is described by an **IB curve**: β (compression
strength) on the x-axis, `I(X;T)` (compression) and `I(T;Y)` (task-relevant information)
on the y-axis. The plateau of the curve is that system's **capacity**. The project
computes and stores this curve for every system (DB2), and uses it for capacity
analysis, surrogate modelling, and cross-family studies.

### 1.7 What this buys in engineering terms

| Benefit | Concrete manifestation |
|---|---|
| **Cross-domain reuse** | Requirement parsing, inversion and verification are **domain-agnostic** (27 families / 52 domains share one mechanism) |
| **Verifiability** | Every criterion maps to a computable quantity, and **anything out of domain returns `unknown`** (no guessing) |
| **Auditability** | Every verification is logged with model, tolerance source, validity domain, known bias, verdict rule |
| **Honesty** | Claims and boundaries fit on one page; negative results are stated; convention mismatches are caught by gates |

---

## 2. System architecture: the current main chain

### 2.1 The picture

```
                     ┌──────────────── five databases ────────────────┐
                     │ DB1 physical systems  DB2 fingerprints         │
                     │ DB3 requirement mappings  DB4 RL experience    │
                     │ DB5 decode verification                        │
                     └────────────────────────────────────────────────┘
requirement text (DB3 anchors)
   │ ①
   ▼
B parser ──(abstain: to human queue)──┐
   │ ② attributes / direction / values  │
   ▼                                    │
reward-type classification (information-quantity / convention / constraint)
   │ ③                                  ← a first-class citizen: it decides the route
   ├── information-quantity ──► q metrics ──► q → parameter inversion (21 families)
   ├── convention ───────────► lookup / convention
   └── constraint ───────────► constraint path (**β does not apply**)
   │ ④
   ▼
design (**state = design vector**) ──► C generator: requirement → design distribution
   5 design domains: two_level / rayleigh / landau / chain / array
   │ ⑤
   ▼
verification API (27 families) ──► pass / fail / **unknown**
   + uncertainty u + distance to boundary + robustness + **certificate**
   │ ⑥
   ▼
feedback loop: reflow → DB5 + weighted dataset → consumed by training
   │ ⑦
   └─► RL (**optional layer**): design-vector-space interface, for non-information objectives only
```

### 2.2 Layer by layer

#### ② B parser (requirement text → q structure)

| Item | Description |
|---|---|
| **Input** | One line of requirement text (mixed Chinese/English) |
| **Output** | Attribute set (e.g. `precision`), direction (`bound_above/below`), confidence; **abstains when unsure** |
| **Implementation** | `b_network/bv1_parse.py` (model), `features_v2.py` (features), `b_network/b_parse.py` (production entry) |
| **Features** | Keywords/units/numbers + character 3-grams and English words + a **rule-predicate-guided bilingual cue block** (derived from `db3/rules/rules.yaml` physical predicates, **not domain labels**) |
| **Abstention** | `τ = 0.50` (chosen from a **risk–coverage curve**) ⟹ abstention means **hand-off to a human**, never a guess |
| **Measured** | In-domain LOOCV **87.0%**; strict LODO (52 domains held out) transductive **85.2%** / inductive **82.8%**; direction 98.5%; **abstention 13–17%** |

#### ③ Reward-type classification (a first-class citizen)

A requirement's *type* decides *which route* it takes:

| Type | Share | Handling |
|---|---|---|
| **Information-quantity** | 46% | Becomes a q-target ⟹ invert physical parameters, calibrate β |
| **Convention** | 33% | Table lookup / standard convention |
| **Constraint** | 21% | **Constraint path** (e.g. barrier/fatigue classes: **β does not apply**) |

Implementation: `pipeline/reward_types.py`; β calibration: `pipeline/beta_calibration.py`.

#### ④ Design (state = design vector)

**Key design decision**: the state is not a graph structure but a **design vector** — one
state can correspond to many designs (non-degenerate).

| Design domain | Dimensionality / type | Notes |
|---|---|---|
| `two_level` | 2 dof (ΔF, T) | one state maps to **200 designs** |
| `rayleigh` | amplitude σ | acoustic/optical amplitude class |
| `landau` | potential parameters | high-dimensional; most mature C instance |
| `chain` | 7-dim (**mixed**) | composite chain (series of physics) |
| `array` | 5-dim (**mixed**) | acoustic array geometry (FFT array factor) |

Implementation: `pipeline/design_space.py` (registry + encode/decode),
`design_contract.py` (decode contract).
**C generator** (requirement → design distribution): `c_network/cfm.py` (conditional flow
matching with exact log-density), `c_network/mdn.py` (domain-specific MDN).

#### ⑤ Verification (three states + certificate)

| Feature | Description |
|---|---|
| **Three states** | `pass` / `fail` / **`unknown`** — out-of-domain, missing target, or adapter failure **always yields `unknown`**, never `pass` |
| **Validity registry** | `sim/registry.json`: 27 families, each with **calibrated** metrics and tolerances |
| **Tolerance** | `max(floor, k/√n)`, with `k` **calibrated against truth**; conservative by roughly **4.6×** |
| **Certificate + uncertainty** | Every verification carries **u (uncertainty)**, **distance to the decision boundary (in units of u)**, **robustness** (robust/marginal), and a **certificate**: model / tolerance source / validity domain / supported metrics / **known bias** / **whether third-party corroboration exists** / **verdict rule** / **reproducible design input** |
| **Audit** | `sim/verify_audit.jsonl` (append-only, one line per verification with evidence and certificate) |

Implementation: `sim/verify_api.py` (API), `sim/expand_registry.py` (calibration and
registration), `sim/simulator_adapter.py` / `rate_adapter.py` / `front_adapter.py`
(adapters).

#### ⑥ Feedback loop

Verification results → **only passing entries are written** → DB5 + a **weighted dataset**
(`reflow_weighted_*.csv`) → consumed by training.
Implementation: `pipeline/design_reflow.py`, `design_reflow_consume.py`.

#### ⑦ RL (optional layer)

**Position**: an interface reserved for **non-information objectives** (cost,
constraints); **state = design vector**. **No performance is claimed** — the measured
ES baseline is worse than random.
Implementation: `pipeline/rl_design_env.py`; the research code lives in `a_network/`
(**off the main chain**).

---

## 3. The five databases (DB1–DB5)

| DB | Contents | Scale | Used by | Red line |
|---|---|---|---|---|
| **DB1** | Physical system entries: domain/family/parameters/equations + **declarative Fisher fingerprint** | **300** | calibration ground truth, β calibration, verification tolerances | machine gate **300/300**; human spot-check **list 60/300 (20%)**, review progress **294 pending / 6 spot_checked**; **AI-generated entries can never be anchors** |
| **DB2** | Information fingerprints: IB curve, Fisher trace, capacity per system | **8879 rows** (**99.7%** usable curves) | inversion, surrogate modelling, cross-family studies | curves and fingerprints must match the truth convention (gated) |
| **DB3** | Requirement-mapping anchors: text ↔ q structure (attributes/direction/values) | **169** | B parser training and evaluation | **requires A/B dual-role human review**; validation 169/169 |
| **DB4** | RL experience (historical layer) | 87 | RL research | historical reference only |
| **DB5** | Decode-verification results (passing entries only) | 390 | feedback loop, weighted training set | must agree with `verdict ∈ {pass, fail}` |

---

## 4. What it can do today

### 4.1 Capability table (each row has a measured figure and a repro command)

| Capability | Measured | Reproduce |
|---|---|---|
| **Requirement text → q structure** | LOOCV **87.0%**; strict LODO inductive **82.8%**; direction **98.5%**; **abstention 13–17% to a human queue** | `python b_network/bv1_train_eval.py`, `python b_network/v31_lodo_check.py` |
| **q → parameter inversion** | **21 families** (three-tier priority: analytic → numerical inversion → an honest `None`) | `python pipeline/inverse_fisher.py` |
| **Rate-type requirements** | `q9`/`q7` **metric routing 51/52**; **capability reachable 51/51** (ou spectral gap, diffusion front, poisson_count/radioactive_decay rates with validity domain `[1e-3, 1e12] s⁻¹`; wavefront `harmonic_oscillator`/`lc_circuit`) | `python sim/v36_rate_coverage.py` |
| **Requirement → design distribution (C)** | **4/4 design domains** (20 seeds each, dual gates): landau 88.4% / rayleigh 96.8% / chain 82.8% / array 76.2% | `python pipeline/v3*_c_check.py` |
| **Design → verification** | **27/27 families** registered; **out-of-domain always `unknown`**; **measurement agrees with truth within 10%** (gated) | `python sim/verify_api.py`, `python sim/registry_truth_scan.py` |
| **Trustworthiness of a verdict** | Every `pass` carries **uncertainty + boundary distance + robustness + certificate** | `python sim/certificate_check.py`, `python tools/demo_minimal.py` |
| **Third-party external corroboration** | **5/27 families**: `diffusion/q7`, `two_level`, `rayleigh`, `weibull`, `landau` (three independent external routes agree; project vs external **0.00%**) | `python sim/external_check_diffusion.py`, `python sim/external_check_five_families.py` |
| **Feedback loop** | reflow → DB5 (390) + weighted dataset → consumed by training (**closed in practice**) | `python pipeline/demo_end_to_end.py --pair DB3-0009` |
| **Whole-chain gate** | fast tier **38 stages = PASS 36 / boundary 2 / known-issue 0 / FAIL 0**; invariant suite **23/23**; core **17/17** | `python run_system.py`, `python tests/test_invariants.py` |

### 4.2 What it explicitly cannot do (with reasons)

| Not usable for | Reason |
|---|---|
| Claiming "the cross-domain unified framework holds" | The analytic side holds; **absolute capacity is not cross-family predictable** (**the theoretical invariant** is — certified at 3.0%). Statements must be **layered**. |
| Treating `PASS` as external endorsement | The verifier is **self-built** ("outside the design loop" ≠ third party); third-party corroboration covers only **5/27 families**. |
| Production embedding | No service API; DB1 human review incomplete; **real requirement text has never been evaluated**. |
| Any RL performance claim | RL is interface and gating only; the measured SAC/v14 gains do not hold. |
| Fully automatic text → q | Slot prediction is automatic; **numeric extraction / rule matching remain semi-automatic**. |
| Citation as experimental fact | Data is **AI-generated + human spot-checked**, **not experimentally authoritative**. |

### 4.3 Negative results and current status

| Item | Status |
|---|---|
| **T1 family-level fingerprint transfer** | ✅ passes |
| **T2 cross-family capacity prediction (absolute)** | ❌ **not passing** (64.0% median): log convention ineffective; profile features bring it to 17.8% but the **interval crosses the threshold, so it is not certified**; adding row-level samples **cannot** tighten the interval |
| **T-c theoretical-invariant cross-family prediction** | ✅ **certified**: predicting `log₁₀(β@90%)` gives a family-level leave-one-out median error of **3.0%**, bootstrap 95% interval **[1.4%, 6.6%]** (caveat: the features include an input-side scale ⟹ the claim is input↔output scale consistency) |
| **M6.2 structural editing** | ✅ **boundary stable** (4.50 / 1.75 / 2.00 / 2.00, reproducible) |
| **SAC single-peak search** | ❌ **73.6% ± 32.7, bistable ⟹ not usable** |
| **v14 structural-space capability curve** | ❌ **40–45% pass rate ⟹ not usable** |
| **Feedback edge** | ❌ **0 positive / 1 harmful / 7 undetermined ⟹ not viable** |
| **Variational-layer L2 increment** | ❌ **no practical increment** on either target quantity |
| **diffusion grid-alignment bias** | ⚠️ known **0.512%** (the probe sits at a node rather than the exact L) — not a failure, but it must be reported alongside any conclusion |
| **Registry measurement vs node truth (all 27 families)** | ✅ **All 27 registered families: measurement agrees with node truth within 10%** — per-family deviation **0.00%–1.32%**; registry calibration **27/27** ✓ (`gaussian_mixture2` / `landau` / `truncated_gaussian`: **0.20% / 0.61% / 0.00%**) — **standing gate** |

> **Methodology** (the most distinctive part of this project): every conclusion follows
> one credibility standard — **≥20 seeds + pre-registered criteria + per-point CSV
> output + three-state verdicts + negative results held to the same standard**;
> "impossible numbers" are treated as bugs; every criterion must self-check whether it
> still discriminates.

---

## 5. Directories and the role of every file

> **Note**: the tables below list **key files** (entry points, criteria, data contracts),
> **not** an exhaustive manifest. Packages also contain `__init__.py`, `db*/**.schema.json`,
> `db*/rules|conventions/`, `tools/*.ps1` and similar supporting files; absence from these
> tables does not mean absence from the repository.

### 5.1 Top level

| Path | Role |
|---|---|
| `README.md` / `README_cn.md` | This document (English / Chinese) |
| `CLAIMS_AND_BOUNDARIES.md` / `_cn.md` | **One-page claims and boundaries** (what it can/cannot do + evidence + mandatory scope statement for citations) |
| `CREDIBILITY.md` | **Credibility index**: the measured status of every headline claim, negative results included (the full internal development log — 214 entries — is not shipped) |
| `cross domain rl architecture plan.md` | Architecture document: the current main chain, the five databases, and the engineering standards |
| `run_system.py` | **End-to-end runner**: 38-stage gate (fast) / +18 heavy stages (`--full`, 56 total); tiers, known-boundary and known-issue classification |
| `run_demo.py` | Demo script (reference implementation) |
| `requirements.txt` | Dependencies: `numpy / scipy / sympy / PyYAML / torch` (no other third-party packages) |
| `LICENSE` | Code **MIT**; data and documents **CC BY 4.0** |
| `*.csv` / `*.jsonl` (root) | **Evidence files**: per-point results for LODO / transfer / feature A-B / τ sweep / external corroboration / design reflow (those cited in CREDIBILITY) |

  **Root evidence-file index** — `vNN` = the Nth numbered verification session, in chronological
  order of the project's development (each producing script's docstring states its question and
  the internal log entry it belongs to); the suffix is semantic:

  | File | What it evidences | Source |
  |---|---|---|
  | `v31_lodo.csv` | Strict leave-one-domain-out (LODO) generalization of the requirement→q parser | `b_network/v31_lodo_check.py` |
  | `v32_full_train_abstain.csv` | Full-train accuracy and abstention routing rates | `b_network/v32_full_train_abstain.py` |
  | `v33_feature_improve.csv` | Fingerprint feature A/B (improvement) | `b_network/v33_feature_improve.py` |
  | `v33_runs.csv` / `v34_runs.csv` / `v35_runs.csv` / `v37_runs.csv` | Per-seed raw runs backing the multi-seed protocols cited in CREDIBILITY | multi-seed re-check |
  | `v34_tau_risk_coverage.csv` | Abstention-threshold τ sweep: risk–coverage trade-off | `b_network/v34_tau_risk_coverage.py` |
  | `v35_new_anchor_transfer.csv` | Transfer to a newly added anchor domain | `b_network/v35_new_anchor_transfer.py` |
  | `v36_capacity_transfer.csv` | T2 negative-result diagnosis (S1+S2): predictability outside the capacity families | `a_network/v36_capacity_transfer.py` |
  | `v37_row_capacity_transfer.csv` | Row-level capacity-family transfer check | `a_network/v37_row_capacity_transfer.py` |
  | `v38_capacity_invariant_transfer.csv` | Capacity-invariant transfer check | `a_network/v38_capacity_invariant_transfer.py` |
  | `v38_rl_interface.csv` | RL environment interface probe (design as actions) | `pipeline/rl_design_env.py` |
  | `registry_truth_scan.csv` | All 27 families: three-state verification vs ground truth (≤10%) | `sim/registry_truth_scan.py` |
  | `external_check_diffusion.csv` | External corroboration, q7 diffusion (three external routes, κ̂ diff 0.00%) | `sim/external_check_diffusion.py` |
  | `external_check_five_families.csv` | External corroboration, 5/27 families | `sim/external_check_five_families.py` |
  | `reflow_weighted_chain.csv` / `_rayleigh.csv` / `_two_level.csv` | Curated verification→design weighted-reflow summaries (closed-loop consumption; consumers: `pipeline/demo_end_to_end.py`, `pipeline/v34_rayleigh_c_check.py`) | reflow aggregation step |

### 5.2 `core/` — information computation core (definition layer)

| File | Role |
|---|---|
| `fisher.py` | **Fisher information**: analytic expressions for 27 families + numerical/Monte-Carlo implementations + a generic finite-difference entry |
| `mutual_info.py` | Mutual-information estimators (KSG k-NN and binning) |
| `ib.py` | **Information bottleneck**: Blahut–Arimoto solver for IB curves (`I_XT`/`I_TY` vs β) |
| `action.py` | **Information Lagrangian**: `ib_potential` (IB functional) and `info_action` (kinetic − potential) |
| `fingerprint.py` | Fingerprint aggregation: node Fisher → trace/determinant/directionality |
| `channels.py` | Interface placeholder for information-structure graph encoding (used by the B side) |
| `tests/test_core.py` | Core self-tests, **17 items** (including Fisher sign/definition regressions) |

### 5.3 `b_network/` — B parser and system encoding (production entry)

| File | Role |
|---|---|
| `b_parse.py` | **Production parsing entry** (unified call: model slots + numeric extraction + rule matching + abstention) |
| `bv1_parse.py` | The neural parser itself: `Bv1Model` (multi-label attributes + direction head), `featurize` (**v2 cue features**), `input_dim`, `Bv1Parser` (with τ abstention and human queue) |
| `features_v2.py` | Feature comparison tool: re-export of production v2 features + **legacy features** (for A/B experiments) |
| `bv1_train_eval.py` | Training and evaluation (LOOCV) + abstention sweep; **writes `b_network/bv1_model.pt`** |
| `bv1_calibrate.py` | Confidence calibration (temperature scaling / Platt); **writes `b_network/bv1_model_cal.pt`** |
| `bv1_dataset.py` | Training-set assembly (DB3 anchors → text / slot labels) |
| `bv1_paraphrase.py` | Language-level augmentation (anchor paraphrasing + grouped LOOCV re-evaluation) |
| `eval_b_testset.py` | Evaluation on a new-wording test set (v0 / v1 / hybrid comparison) |
| `smoke_b_parse.py` | Smoke test for the production entry |
| `system_encoder.py` | **System → information-structure graph** (`EncodedGraph`: X→T→Y nodes and edges; IB source registry) |
| `fingerprint_computer.py` | **Fingerprint computation**: structure graph → fingerprint (numeric vs analytic cross-check, IB curves) |
| `metric_spec.py` | Requirement-side core data class (`MetricSpec`) |
| `requirement_parser.py` | v0 rule parser: text → structured requirement |
| `rule_matcher.py` | Structured requirement → matched rules |
| `composer.py` | Matched rules → `MetricSpec` (rule-line assembly) |
| `key_registry.py` | Canonical key registry (**single source of truth**, prevents key drift) |
| `v31_lodo_check.py` | **Strict leave-one-domain-out check** (52 domains; transductive and inductive) |
| `v31b_fail_domain_diag.py` | Failure-mode localisation for fully-failing domains (out-of-vocabulary / labelling / sample size) |
| `v32_full_train_abstain.py` | Full training + abstention comparison (two arms + LOPO) |
| `v33_feature_improve.py` | Feature A/B (v1 vs v2 cue block, three protocols) |
| `v34_tau_risk_coverage.py` | **Risk–coverage curve** for the abstention threshold τ |
| `v35_new_anchor_transfer.py` | New-anchor **in-domain transfer** re-test (standing stage) |
| `test_new_encoders*.py` (9 files) | Encoder self-tests, one per family batch |
| `test_ib_nonparametric.py` / `test_theory_constraints.py` / `test_encoder.py` / `test_new_families.py` | Non-parametric IB / theory constraints / encoder / family-extension Monte-Carlo verification |

### 5.4 `pipeline/` — engineering layer (mid-chain)

| File | Role |
|---|---|
| `reward_types.py` | **Reward-type classification (first-class citizen)**: information-quantity / convention / constraint |
| `beta_calibration.py` | **Requirement → β calibration** (replaces placeholders with derivable values; not applicable to constraints) |
| `inverse_fisher.py` | **q → parameter inversion** (21 families; analytic → numerical → honest `None`) |
| `design_space.py` | **Design-space registry**: 5 domains, state = design vector, `encode/decode_design` |
| `design_contract.py` | Design decode contract + requirement inversion (closes the breakpoints) |
| `design_two_level.py` / `design_rayleigh.py` / `design_landau.py` / `design_chain.py` / `design_array.py` | Per-domain **pre-checks and non-degeneracy verification** (`array` includes FFT array factor and grating-lobe conditions) |
| `v33_landau_c_check.py` / `v34_rayleigh_c_check.py` / `v35_chain_c_check.py` / `v37_array_c_check.py` | **C instances**: requirement → design distribution (20 seeds, dual gates, per-seed output) |
| `design_reflow.py` / `design_reflow_consume.py` | **Verification feedback loop**: passing entries only → DB5 + weighted dataset |
| `family_aliases.py` | Family-name aliases (**single source of truth**) |
| `extra_curves.py` | Barrier curves and operating-condition index curves |
| `rl_design_env.py` | **RL interface (optional layer)**: design-vector-space environment + gates |
| `run_loop.py` | Functional end-to-end loop (with breakpoint report) |
| `demo_end_to_end.py` | **Single-requirement full-chain demo** (parse → classify → invert → design → verify → feed back) |

### 5.5 `sim/` — verification (outside the design loop + third-party corroboration)

| File | Role |
|---|---|
| `verify_api.py` | **Production verification API**: three-state verdict + **uncertainty/certificate**; out-of-domain always `unknown` |
| `expand_registry.py` | **Calibration and registration**: derives samplers from encoders, calibrates each family against truth (registered only if p99 ≤ 10%) |
| `registry_truth_scan.py` | **Gate**: per-family check that "registry measurement vs node truth" agrees (≤10%) |
| `certificate_check.py` | **Pre-registered regression checks** for certificates and uncertainty (H1–H5: verdict semantics unchanged, fields complete, audit append-only, …) |
| `simulator_adapter.py` | Simulator adapters: analytic families + dynamics (e.g. heat-conduction half-rise time) |
| `rate_adapter.py` / `front_adapter.py` | Rate-type / front-type adapters (`q9`/`q7` routes) |
| `v36_rate_coverage.py` | **Metric routing and capability reachability** for rate-type requirements |
| `sim_check.py` / `verify_designs.py` | Simulator truth self-check / analytic verdict vs simulated verdict |
| `external_check_diffusion.py` | **Third-party corroboration**: diffusion/q7 via three external methods (analytic series / Crank–Nicolson / Monte-Carlo) |
| `external_check_five_families.py` | **Third-party corroboration extended to 5 families** (closed form / numerical differentiation and integration / Monte-Carlo score-square) |

### 5.6 `c_network/` — C generator (generation side of the main chain)

| File | Role |
|---|---|
| `cfm.py` | Conditional flow-matching decoder (with **exact log-density**) |
| `mdn.py` | Domain-specific MDN decoder (landau instance) |
| `v22_cfm_check.py` / `v23_cfm_selfcheck.py` / `v24_mdn_symmetry_check.py` | Coverage audit / instrument self-check / symmetry measurement (**prove the ruler is accurate before judging**) |
| `v8_mdn_check.py` / `v12_c_feedback_check.py` / `v16_plasma_check.py` | MDN criterion / feedback edge / domain instance verification |

### 5.7 `a_network/` — A-side representation and RL research (**off the main chain**)

| Group | Files | Role |
|---|---|---|
| **Still called by the main chain** | `analytic_backbone.py` | Analytic backbone (fast-path analytic quantity computation) |
| | `action_reward.py` | Action reward (`β·I_TY − I_XT` semantics) |
| | `ib_surrogate.py` | IB-curve surrogate (assembles datasets from DB2, trains a monotone surrogate) |
| **Representation** | `gine.py` / `heads.py` / `struct_embed.py` / `graph_dataset.py` | GINE encoder / six read-out heads / structural embedding / graph tensorisation |
| **RL and search** | `gflownet.py` / `sac.py` / `cma_es.py` / `graph_env.py` / `structure_env.py` / `req_env.py` / `rl_env.py` / `edit_pipeline.py` | GFlowNet / SAC / CMA-ES / graph-edit and structure environments / edit-effect pipeline |
| **Data and feedback** | `db5_feedback.py` / `offline_db4.py` / `augment.py` | DB5 feedback / DB4 off-policy wiring / augmentation |
| **Multi-seed checks and diagnosis** | `multi_seed_verify.py` / `v*_check.py` (a dozen) / `diagnose_landau_*.py` | Multi-seed checks / per-version capability checks / landau diagnosis |
| **Cross-family studies** | `v29_cross_domain_transfer.py` / `v36_capacity_transfer.py` / `v37_row_capacity_transfer.py` / `v38_capacity_invariant_transfer.py` | T1–T3 transfer / T2 diagnosis / row-level re-test / **T-c invariant certification** |

> This is **research code**: the main chain does not depend on it (except the three
> "still called" modules above).

### 5.8 `db1/`–`db5/` — building and maintaining the five databases

| DB | Key files | Role |
|---|---|---|
| **DB1** | `validate_entry.py` | **Machine gate** (schema / conventions / computability / analytic cross-check) |
| | `review_queue.py` | **Human review queue** (`next` / `brief` / `worksheet` / `apply`; A/B dual roles) |
| | `independent_check.py` | **Independent numerical check** (numerically differentiate the Fisher definition without calling the truth function) |
| | `next_batch.py` / `batch_generate_ingest.py` / `ingest_pipeline.py` / `ingest_sim_data.py` | Batch generation and ingestion / simulation-data ingestion |
| | `mode3_generate.py` / `mode3_llm.py` / `gen_landau_deep_entries.py` | Entry generation (template / deep scenarios) |
| | `migrate_note_to_source_ref.py` / `remove_systems.py` | Data migration and cleanup |
| | `db2_writer.py` | DB1 → DB2 fingerprint writing |
| **DB2** | `estimate_fingerprint.py` / `fill_empty_curves.py` / `rebuild_family_curves.py` / `unify_by_family.py` | Fingerprint estimation / curve filling / rebuilding / per-family unification (each with invariant re-checks) |
| | `attribute_fill_effect.py` / `m4_variance_check.py` | **Attribution experiment** for data changes and criterion **variance check** |
| **DB3** | `validate_pair.py` / `review_queue.py` / `batch_review.py` | Anchor validation / review queue / batch review (A+B) |
| | `check_parser_consistency.py` | **Anchor ↔ parser consistency** (gate: error count must be 0) |
| | `generate_pairs.py` / `gen_pairs_batch.py` / `propose_anchors.py` | Anchor generation (table-driven) / **candidate drafting** |
| | `dedupe_pairs.py` / `migrate_direction_semantics.py` / `post_review_fixes.py` / `confirm_quantity_bounds.py` / `roundtrip_check.py` / `repair_pairs.py` | Deduplication / direction-semantics migration / post-review fixes / quantifier confirmation / round-trip check / **exact repair tool** |
| **DB4** | `v9_db4_build.py` / `db4_writer.py` | RL-experience database build and writing |
| **DB5** | `v10_db5_build.py` / `db5_writer.py` / `demo_db5_analytic.py` | Decode-verification database build / writing / demo |

### 5.9 `tests/` and `tools/`

| File | Role |
|---|---|
| `tests/test_invariants.py` | **Core invariant suite, 23 items**: covers the known defect classes (quantity/convention mismatch, caching, indexing, sign errors, normalisation, …); pytest-compatible |
| `tools/demo_minimal.py` | **Minimal verification demo**: requirement → q → parameter bounds → design → three-state verification → external corroboration (~10 s) |
| `tools/privacy_scan.py` | **Privacy/secret scanner**: 13 pre-registered rules (keys / private keys / emails / phone numbers / home directories / environment paths + credential-like filenames); **non-zero exit on any hit** |
| `tools/make_release_copy.py` | **Release-copy tooling**: **generate** the release copy (explicit manifest + **mirror prune** + automatic privacy scan) and **verify** it with `--verify` — per-file SHA256 check that nothing is missing, nothing differs and nothing is extra |
| `tools/db1_checklist.py` | Generates the DB1 human-review checklist (internal review artifact, not shipped) |
| `tools/tidy_for_release.py` | Pre-release tidy-up (**explicit list, dry-run by default, move-never-delete**) |

### 5.10 `md/` — self-written theory notes (**reference reading, not project output**)

**17 notes** shipped in this release: the derivation chain and delimitation
(`information mechanics derivations.md`, `information mechanics delimitation.md`,
`appendix derivations.md`), theory ↔ AI mappings (`physical to ai*.md`, `ai to physical.md`),
and topical notes (information causality, information analysis, information time,
information quantum gravity, mass in information mechanics, particle information,
physical information, `physical_information_theory_notes.md`, `theory ai.md`,
`why these limits.md`).

> **These are reference reading, not project output**, and they are not needed to run any
> stage.
> **Third-party book/paper notes are not part of this repository** (copyright): a larger
> set of notes — including textbook summaries — exists only in the author's private
> working tree and is deliberately excluded here.

### 5.11 Working language

Code comments, docstrings, and the internal development log are written **primarily in
Chinese** (the project's working language). English translations are provided for the
reader-facing documents listed in §5.1. Note that **Chinese string literals inside
`b_network/` are intentional**, not untranslated text: the requirement parser is designed
to handle mixed Chinese/English requirement text.

---

## 6. Quick start

```powershell
# 0) Dependencies (no other third-party packages)
pip install -r requirements.txt

# 1) Whole-chain gate (fast tier, ~6–8 minutes)
python run_system.py            # expect: PASS 36 / boundary 2 / known-issue 0 / FAIL 0 (38 stages)
python run_system.py --full     # adds 18 heavy stages (training / multi-seed / strict LODO / external corroboration / certificates)
python run_system.py --list     # show the stage table only

# 2) Core invariant suite (seconds, 23 items)
python tests/test_invariants.py

# 3) One requirement through the full chain (recommended first look)
python pipeline/demo_end_to_end.py --pair DB3-0009

# 4) Minimal verification demo (with uncertainty and certificate)
python tools/demo_minimal.py

# 5) Call the verification API directly
python -c "from sim.verify_api import verify; v=verify({'family':'rayleigh','sigma':0.5,'metric':'q1','target':3.0,'direction':'lower_bound'}); print(v.verdict, v.measured, v.u, v.robustness, v.certificate['verdict_rule'])"
```

**Expected output** (a `pass` example):

```
verdict=pass | measured=15.81 | u=0.0849 | distance to boundary +56.07 u ⟹ robust
certificate: model=analytic Fisher (family=rayleigh, metric=q1) | tolerance source=… | validity domain={'sigma':[0.1,5.0]}
             known bias=none recorded | external corroboration=yes (three external routes agree) | verdict rule=measured ≥ target·(1−tol)
```

### Model weights (**shipped in this repository**)

The B parser needs two weight files, and they **are part of the repository**, so a
**fresh clone already has the full parser** with no training required:

| File | Size | Role |
|---|---|---|
| `b_network/bv1_model.pt` | 0.33 MB | Neural slot model (attribute set + direction head) |
| `b_network/bv1_model_cal.pt` | 0.31 MB | **Calibrated** confidence version (Platt/temperature, used for abstention) |

**If you want to retrain / recalibrate** (e.g. after changing features or adding DB3 anchors):

```powershell
# Retrain the slot model (LOOCV evaluation + full training, saves b_network/bv1_model.pt)
python b_network/bv1_train_eval.py

# Recalibrate confidence (writes b_network/bv1_model_cal.pt)
python b_network/bv1_calibrate.py
```

**What happens if the weights are missing** (by design): `b_parse` does **not** raise; it
falls back gracefully to the v0 rule parser and **labels the status explicitly**:

```python
r = parse("The actuator must reach 95% of its travel within 20 ms of a step command.")
r.status      # "auto" (neural) | "human_queue" (abstained) | "v0_fallback" (model missing)
```

> ⚠️ **Convention note**: the fallback loses a lot of capability — on the same
> **new-wording test set**, v1 scores **74.1% / 80.0%** (attribute / direction) while v0
> scores **59.3% / 60.0%** (same protocol, directly comparable). The **87.0%** quoted above
> is **in-domain LOOCV over 169 anchors**, a **different protocol — do not mix the two**. So
> if you see `v0_fallback`, the weights did not load.
> Also, the production abstention threshold is the **explicit τ = 0.50 in code**; the τ
> recorded inside the checkpoint is for reference only.

---

## 7. Data red lines (must read)

1. **The overwhelming majority of data entries here are AI-generated** (template filling
   + machine gate), with human-review status in each entry's
   `provenance.human_review_status` (`pending` / `spot_checked` / `fully_reviewed`).
2. **AI-generated entries are never used as anchors** (`anchor_eligible` is always
   `false`); DB3 requirement anchors require **A/B dual-role human review**.
3. **Do not cite the numbers in this repository as experimental facts.** Every headline
   conclusion is annotated with its seed count / protocol / citability (see the
   credibility index in `CREDIBILITY.md`).
4. **Do not delete** the data-nature statements in `README.md` /
   `CLAIMS_AND_BOUNDARIES.md` / `LICENSE`.

---

## 8. License, citation, further reading

- **Code**: [MIT](LICENSE) ｜ **Data and documents**: CC BY 4.0 (see the last section of `LICENSE`)
- **Citation**: if you use this repository's data or methods, please cite the repository
  and respect the **citation status** annotated on each item.
- **Further reading**:
  - [`CLAIMS_AND_BOUNDARIES.md`](CLAIMS_AND_BOUNDARIES.md) ｜ [中文](CLAIMS_AND_BOUNDARIES_cn.md) — what it can/cannot do, and the scope statement any citation must carry
  - [`CREDIBILITY.md`](CREDIBILITY.md) — credibility index: the measured status of every headline claim
  - [`cross domain rl architecture plan.md`](cross%20domain%20rl%20architecture%20plan.md) — architecture: the current main chain, the five databases, and the engineering standards
