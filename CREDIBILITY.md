> **English** ｜ 中文 (full internal development log, not shipped)

# Handover — Credibility Index

**This is the citable credibility index for the project.** Every headline claim is listed with its protocol
(seed count, corpus size, data version) and the status a citation must carry — what is established, what is
bounded, and what must never be quoted as a result. The complete internal development log (214 numbered
entries) is kept outside the published set; this file is the citable summary of it.

---

## 1. What this file is

Two properties matter more than the volume:

1. **Every headline result is annotated with its credibility status.** Each claim carries its protocol (seed
   count, corpus size, data version) and whether it may be quoted as a conclusion.
2. **Negative results are stated at the same weight as successes.** Where a measurement contradicts a claim,
   the measurement is what stands; criteria and thresholds are never moved to make a result pass.

This index reflects the project's data and measurements as of **2026-10-02**.

---

## 2. The project in one paragraph

"Information mechanics" (Data Processing Inequality as sole axiom → Fisher metric → information
Lagrangian → projection of physical laws) is landed as a **cross-domain unified design system**:
a three-network architecture (A = general physics knowledge learner / B = encoder / C = decoder) plus an
RL framework, driven by a five-library data system (DB1–DB5).

The **main chain is C-dominant: RL is an optional layer and the state is the design vector**. The A network is
**not on the main chain** (it remains as research code). The main chain:

① requirement (DB3 anchors) → ② B parser (abstention → human queue) → ③ reward-type classification
(information-quantity / convention / constraint) → ④ 5 design domains + C generation (4/4 domains) →
⑤ three-state verification + certificate/uncertainty → ⑥ feedback loop (substantively closed) →
⑦ RL (optional layer, **no performance claim**).

Positions worth knowing before citing this project:

- The project **does not claim** DPI, Fisher uniqueness, the Information Bottleneck or information
  geometry as its own inventions; the claimed novelty is the **re-assembly** around DPI as sole axiom,
  the **engineering dictionary** from physical systems to information-structure graphs, and the
  cross-domain transfer experiments.
- The philosophical stance stated in the README — "a physical system is essentially a
  signal-processing system, still bound by information theory; information carried by a physical field
  cannot propagate in a way that violates information theory, which indirectly constrains the field's
  evolution" — is labelled a **guiding judgement, not a proven theorem**; the project does **not** claim
  to have derived new physics from information theory.
- **RL conclusions require ≥20 seeds + pre-registered criteria** before they count as conclusions;
  numbers resting on **1–12 seeds** are leads, not conclusions.

---

## 3. Credibility index — headline claims and their status

**Status vocabulary**: ✅ **citable** = may be quoted as a conclusion, with its protocol stated;
⚠️ = quote only with the stated limitation attached; ❌ = not achieved — do not cite as a result.

### 3.1 B parser (requirement text → q structure) — the strongest line

| Claim | Value as recorded | Protocol / scale | Status |
|---|---|---|---|
| **LOOCV (in-domain)** | **87.0%** (147/169); unmapped 98.2%; direction 95.8% | 169 DB3 anchors | ✅ citable |
| **Strict LODO, transductive** | **85.2%** (abstention 13.0%, unmapped 88.2%, direction 98.7%) | leave-one-domain-out, 52 domains | ✅ citable |
| **Strict LODO, inductive** | **82.8%** (abstention 17.2%, direction **98.5%**); criterion L5 "cross-domain generalisation holds" (≥70%) | 52 domains, vocabulary built from training domains only ⟹ no vocabulary leakage | ✅ citable |
| **Abstention rate** | **13–17%** (13.0% transductive / 17.2% inductive) | production semantics (`conf < 0.90` ⟹ abstain) | ✅ citable |
| In-domain transfer of newly added anchors (leave-one-out) | v2 **5/10** ⟹ holds (criterion ≥5) | 10 newest anchors | ✅ citable |
| τ sweep (LOOCV) | τ=0.50 ⟹ coverage 89%, precision 94.0% | — | ✅ citable |
| **Production configuration (feature-v2 + τ=0.50)** | LODO inductive **87.4%**; transductive **83.0%**; abstention **12.6%**; LOOCV **88.1%**; feature dim **2167** | 159-anchor corpus ⟹ **state the corpus size** — the 169-anchor corpus gives 87.0 / 85.2 / 82.8 | ✅ citable |
| **LODO under a second scoring convention** | transductive **84.9%**, inductive **87.4%**, unmapped **97.5%**, direction **97.4%** | 159 anchors | ⚠️ **two scoring conventions are on record — always quote the corpus and the convention together** |
| Production posture | τ=0.9 ⟹ 81% auto @96.9% precision; consensus 42% @92.9% precision / direction 100% | 159 anchors + 67 held-out | ✅ citable |

### 3.2 Theory, cross-domain transfer, verification

| Claim | Value as recorded | Protocol / scale | Status |
|---|---|---|---|
| **Inverse (q → parameter) table** | **21 families invertible** (target ≥10); **5 non-monotone families flagged** (`beta`/`frechet`/`gamma`/`two_level`/`weibull` — exactly the theoretically expected ones); unreachable 1 (`gaussian_mixture2`); same-convention check **0.07%**; formula-level check **0.00%** | 27 families × 48-point grid; triple check 3/3 | ✅ citable; **limited to q1 (Fisher-type)** |
| **Verification registry coverage** | **27/27 families** registered (out-of-domain ⟹ `unknown`, never `pass`) | `sim/registry.json` | ✅ citable |
| **Registry measurement vs node truth** | All 27 families: per-family deviation **0.00%–1.32%** — highest `vonmises` **1.32%**; `gaussian_mixture2` **0.20%**, `landau` **0.61%**, `truncated_gaussian` **0.00%**; control `truncated_geometric` **0.95%**; **0/27 families over the ≤10% limit** | pre-registered R1 ≤10%; **standing gate** | ✅ citable |
| **External (third-party-style) corroboration** | **5/27 families** (`diffusion`/q7 + `two_level`/`rayleigh`/`weibull`/`landau`); G1 A↔B 0.00% (12/12 rows), G3 project-vs-external ≤10% with max 0.00%; for diffusion: E1 0.002%, E2 within ±3σ, E3 0.00% | pre-registered G1–G4 / E1–E4 | ✅ citable, **with the project's own scope caveat**: the "external" methods are implemented in-project from textbooks/statistics principles, **not third-party software**; the other 22 families remain internally self-consistent only |
| **T-c theoretical invariant** | saturation scale `log10(β@90%)` predicted across families to **3.0%**, bootstrap 95% CI **[1.4%, 6.6%]** (null model 23.1%) ⟹ **certified**; E4: channel-efficiency ratio Y1 ∈ [0.0154, 0.5553] ⊆ (0,1] ⟹ DPI numerically confirmed on 27 families | 27 families, pre-registered E0–E5 | ✅ citable, **with reservation**: prediction features include the input-side saturation scale ⟹ the proposition is "input↔output scale consistency", **not** prediction from zero |
| **T2 absolute capacity** | **NOT passing**: v29 `raw_base` median relative error **64.0%** (mean 110.0%) against a ≤20% criterion; `log_base` 38.3%; main arm `log_rich` 28.2% (still >20%); row-level arm 62.3% (69.4% family-level), bootstrap [44.9%, 107.9%] | 27 families; criteria unchanged | ❌ not citable as achieved — **recorded as not meeting** |
| Y1 / Y3 secondary targets | Y1 22.5%, CI [8.3%, 41.8%] crosses threshold; Y3 8.5% does **not beat** the null model (7.1%) | 27 families | ⚠️ negative / not certified |
| Gap ③: composite structure under an information objective | **has no value** — evaluable rate 100% / 38% / 12% / 10% (1–4 nodes); 3-node `I_TY` median **1.15e-07**; satisfaction **0/7** | 60 structures per size | ✅ citable; theorem-level (DPI ⟹ longer chain ⟹ less information); gap ③ is classified "will not do" |
| Phase 4 cross-domain transfer | representation layer ✅ (gap median **4.2%**, per-family ranking differs 13×); **family-level capacity ❌ not extrapolable (error 96%)**; inverse rule ✅ family-independent | 13 families / 432 rows | ✅ citable; **domain-level transfer not done** |
| Non-degenerate design domain | `two_level` design `(ΔF, T)` = 2 dof ⟹ **200 designs per state**; closed loop through the existing encoder gives q1 relative difference **0.0e+00**, DPI-legal | 1 family | ✅ citable; **the only non-degenerate domain of its kind** |
| Requirement → β calibration | **109/159 calibratable** (β ∈ 0.754–5.612, median 1.507); 109 = information-quantity-type **57** + rate-type convention **52**; the genuinely "information-quantity" share is **46% (73/159)**, rate-type 33%, constraint-type 21% (no β); monotonicity 99.0% | 159 anchors | ✅ citable **only for "information-quantity type"**; constraint-type uses an attainment reward; rate-type is marked a **convention** |
| Variational layer | L1 analytic comparison **all pass** (incl. geodesic on the scale family `θ(t)=4^t`, error ≤0.23%); **L2: no increment on either target** (Δ = −0.0018, 0/20; ρ = −0.009±0.052, 0/20) | 3 cases / 20+20 seeds | ✅ L1 citable; **L2 limited to "no practical increment"** |
| q7 operational-convention factor | **0.9095 = 4·erfcinv(0.5)²** | analytic + numeric | ✅ citable (registered in `q_definitions`) |
| Verification certificates | H1–H5 all pass; robustness distribution robust 6 / marginal 0 / n/a 2 (pass 3 / fail 3 / unknown 2, n=8); the audit log is append-only and holds **1345** lines with **0** lines failing to parse | pre-registered H1–H5 | ✅ citable |

### 3.3 C side (generation)

| Claim | Value as recorded | Protocol / scale | Status |
|---|---|---|---|
| **C instances per domain** | **4/4 domains**: `landau` **88.4% ± 3.0%**, `rayleigh` **96.8% ± 1.3%**, `chain` **82.8% ± 2.8%**, `array` **76.2% ± 1.9%** | 20 seeds each, dual gate C1 (≥70% criterion, unchanged throughout) + C2 diversity 160, all pass | ✅ citable |
| C-side model selection (MDN vs CFM) | MDN coverage **0.9663** (double-well recall 0.950, extrapolation 0.9691) vs CFM coverage **0.9539** (double-well recall 0.975, extrapolation 0.9562, NLL +0.003) ⟹ **both ≫90%, the switch trigger does not fire**; each wins in some domains | 20 + 20 seeds, comparable protocol | ✅ citable |
| Rate/transport capability reachability | q7 capability reachable **51/51 (100%)**; routing 51/52 (1 requirement has no metric registration) | registry capability intervals | ✅ citable |

### 3.4 Databases, gates and suites (measured state)

| Item | Value as recorded | Notes / caveats |
|---|---|---|
| **DB1** physical-system entries | **300** systems; machine gate **300/300** | spot-check list **60/300 (exactly 20%)** — a *list size*, not a pass rate. Independent numeric cross-check: **122 consistent / 2 attention / 176 no conclusion** (35 no declarative form + 29 method failed self-calibration + 112 convention not comparable); **0 real anomalies**. Human review progress: **294 pending / 6 spot_checked** ⟹ **the large majority of DB1 is still awaiting human review, and no AI may review it** |
| **DB2** information fingerprints | **8879** rows; usable curves **8854/8879 = 99.7%** | The production version is the **"fill-only"** version — the variant that passes both invariants simultaneously (M4 proxy gate PASS with Spearman 0.951 **and** β-calibratable count 109). **25 rows are empty** because the channel genuinely carries zero information for those parameters. Convention unification rate **26/27 families** (`diffusion` is deliberately left unified-out: unifying it drops β-calibratable 109 → 88). Residual mixed curve provenance remains for a few families (`ou` 96/390, `diffusion`) ⟹ **"fully unified convention" does not hold** |
| **DB3** requirement-mapping anchors | **169** pairs, **52 domains**, all approved | provenance: AI-drafted + human-confirmed; adding 1–2 anchors **cannot** teach the parser a domain (LOPO transfer 1/10) |
| **DB4** RL experience | **87** structural experiences (GFN 20 / SAC 3 / random 63 / domain instance 1) | historical layer |
| **DB5** decode validation | **390** rows | simulator evidence carries margin policy and operational-convention metadata; `unknown ≠ pass` |
| Design domains | **5** registered domains (state/action dimensions 2/2/3/7/5) + C instances in **4/4** domains | RL interface gate: 5 domains × 20 seeds = 100 runs; R1/R2/R4 pass; **R3: uniform random mean 0.6962 vs (1+1)-ES 0.6444 (difference −0.0518)** — ES wins in 70% of single runs but its mean is dragged down by the `chain` domain |
| Runner / gates | **total 56 stages = fast 38 + full-only 18**; **fast tier 38 stages = PASS 36 / boundary 2 / known-issue 0 / FAIL 0**; invariant suite **23/23**; core self-test **17/17** | boundary stages are *expected* failures, see §6 |
| M5 GFlowNet bimodal sampling | **+1.5 mode 0.372** / **−1.5 mode 0.361** (std 0.028 / 0.029) | **20 seeds, robust (20/20)** ⟹ citable |
| M6.2 structure editing | four criteria **0/20** (`landau` / `elastic` / family-switch / termination); ΔlogR **+1.45±0.76**, 17/20 | **20 seeds; robustly failed** ⟹ the failure is a *stable* boundary, not single-seed luck. Boundary reproduction: M1 **4.50%** / M2 **1.75%** / M3 ΔlogR **1.18** PASS / M4 **2.00%** FAIL / M5 **2.00%** FAIL — identical on re-run |
| landau family IB curve | 270 rows, **28 usable curves**; dip **1.9e-06**, below the other families' **0.00466** | stored append-only |
| Gap ① external validation | agreement **50/52 (96%)**, decidable region **100%**; truth self-check 7/7 | 52 designs |

---

## 4. Negative results and current boundaries

Limits that hold today, with the figures that establish them.

### 4.1 Not usable / not reproducible

1. **SAC single-peak optimisation (M7): not usable.** Mean **73.6%**, std **32.7**, range 0.152…1.000; **13/20**
   seeds meet the criterion (65%) ⟹ **unstable (seed-dependent)**, with **bistable convergence** (7 seeds at
   1.000, 4 seeds below 0.40). No single-peak search figure may be quoted as a capability.
2. **v14 structural-space capability curve: not usable.** Across three σ the means are **+0.72 / +0.74 / +0.56**
   (std ≈1.0), attainment **40–45%** (8/20, 9/20, 8/20).
3. **Feedback edge (DB5 → RL): not viable.** **0 positive / 1 harmful / 7 undetermined** ⟹ the edge is an
   **optional path** in the architecture, not a closed loop.
4. **RL baseline: the ES baseline is worse than random on the mean.** Uniform-random mean **0.6962** vs
   (1+1)-ES **0.6444** (difference **−0.0518**), even though ES wins in 70% of individual runs; the mean is
   dragged down by the `chain` domain. ⟹ **no RL performance claim is made.**
5. **Variational layer L2 — no practical increment.** Δ = **−0.0018** (0/20) and ρ = **−0.009±0.052** (0/20) on
   the two target quantities; the conclusion is limited to "no practical increment".
6. **`T2` as measured today: 64.0% median** (numbers measured before the DB2 curve fill, 41.5% → 99.7%, **are not
   reproducible** and must not be quoted as current).
7. **Diffusion/q7 known grid-alignment bias: 0.512%.** The probe node sits at `x = 0.009975`, i.e. **0.251%**
   closer than `L`, which biases `t_half` by **0.512%** and κ̂ correspondingly by **0.512%**. This bias must be
   reported alongside any diffusion/q7 conclusion (grid: `nx=401`, domain 3L ⟹ `L/spacing = 133.33` does not
   land on a grid node).
8. **Gap ③ (composite structure) is a theorem-level negative**, not a failed experiment: 3-node `I_TY` median
   **1.15e-07**, satisfaction **0/7** ⟹ classified "will not do".
9. **Domain-internal transfer is limited.** Leave-one-out transfer of newly added anchors is **5/10** (criterion
   ≥5); adding 1–2 anchors **cannot** teach the parser a domain (LOPO record: **1/10**), and the two figures use
   **different protocols — do not mix them**.

### 4.2 Standing gates and citation rules

- **Registry measurement vs node truth (standing gate).** All 27 registered families agree with node truth within
  **10%** — per-family deviation **0.00%–1.32%** (highest `vonmises` **1.32%**). `verify_api` values and verdicts
  for `gaussian_mixture2`, `landau` and `truncated_gaussian` must come from the current registry
  (**0.20% / 0.61% / 0.00%**).
- **DB2 curve coverage is 99.7% (8854/8879), and the production version is "fill-only"** — the variant that
  passes both invariants (M4 proxy gate PASS with Spearman **0.951** and β-calibratable count **109**). M4 is
  **deterministic** (σ = **0.000**, 4/4 identical verdicts). ⟹ **coverage alone is not quality**: data changes
  are verified together with the surrogate and the β invariant.
- **The verification audit log is append-only** (`sim/verify_audit.jsonl`): every verification appends one line
  with its evidence and certificate, and no line is rewritten.

### 4.3 Scope limits

- **External corroboration covers 5/27 families.** Claiming that "the whole verification suite has been
  externally validated" is forbidden.
- **The verifier is self-built**: "outside the design loop" is not "third-party independent", so a `PASS` must not
  be read as external endorsement.
- **T2 absolute capacity does not pass; real free-text requirements are untested; RL makes no performance claim;
  the data is not experimental authority.**

---

## 5. Data red lines (violate ⟹ the change is rejected)

**Provenance and anchors**

- AI-generated data may only be labelled `source = AI+verified` (never `textbook`); `verified_by` lists
  **only verification actually performed** (the machine gate rejects `human_review` if no human review
  happened); `anchor_formulas` must be an empty array; review records carry `anchor_eligible` **always
  `false`** and `human_review_status` starting at `pending`.
- **AI-generated content may not be used as a DB3 anchor** until it passes **A+B human review**. The
  three generation modes — improvised in conversation (①), hard-coded at authoring time (②, the current
  mode), and a runtime LLM API loop (③, not built) — are **epistemically equivalent** (all are LLM
  parameterised knowledge) and equally ineligible.
- Simulated evidence is labelled `AI+simulated` and carries margin policy plus operational-convention
  metadata. `unknown ≠ pass`: out-of-domain results are reported as `unknown`, never as `pass`.

**Human review**

- Reviewer A (physics) spot-checks **≥20%** → path endpoint DB2 (status `spot_checked`); A **and** B both
  accept → `fully_reviewed` → only then is an entry eligible as a DB3 anchor candidate.
- A failed spot-check sends the **whole batch back** — local patching is rejected because it
  hides systematic bias.
- **AI must not self-review DB1 or DB3.** The AI may only supply evidence (machine gates, independent
  numeric cross-checks, briefs/worksheets); `human_review_status` is written by the human `apply`
  command alone. The review brief makes **no judgement**.

**Never move the goalposts**

- Criteria are **pre-registered and stay fixed** when a result is bad: the C1 threshold is **70%**; data changes
  are reverted **without touching any criterion**; and a pre-registered criterion that failed (feature-v2 at
  τ=0.90 needing **+5 pp**, delivering **+3.8 pp**) is recorded as a **negative result**; the τ change, not the
  feature change, is the real lever.
- Numbers that have not been re-checked are labelled **"not re-checked"**; RL-side conclusions need **≥20 seeds +
  pre-registered criteria**; the three-state verdict is ≥70% robustly holds / ≤30% robustly fails /
  anything else **unstable**; **negative conclusions are re-checked too**.
- Non-expected FAILs are never treated as results: they trigger a rollback, not a re-description.

**Deletions and moves**

- Deletions/moves require an **explicit ID or file list**, a **dry run by default**, and a **printed
  impact statement**. The tools follow this: `tools/tidy_for_release.py`,
  `tools/make_release_copy.py` and `tools/prune` are explicit-list, dry-run-by-default, and
  **move/copy-only (never delete)**; the prune step has three safety gates (only inside the target; the
  target must look like this project's copy; every item is printed) and deletes nothing outside the
  target.
- Wildcard deletion of data files is forbidden; never assume a CLI argument took effect — dry-run and
  check the actual impact surface first.
- Convention defences: information quantities are always in **nats (natural log), never bits**; Gaussian
  parameterisation uses **`sigma`** (variance written `sigma^2`); SDEs default to **Itô**; the heat bath
  defaults to `k_B T = 1`; bridge constants must carry a `unit` and `is_bridge_constant: true`;
  **edge direction = information-flow direction `X → T → Y`** with the DPI chain `I(X;T1) ≥ I(X;T2) ≥ …`.

---

## 6. How to use this file

1. **Before citing any number, check its status here first.** ✅ **citable** = quote it with its protocol;
   ⚠️ = quote it only with the stated limitation; ❌ = not achieved, never cite it as a result.
2. **Treat `known-issue` and `boundary` stage labels as first-class information** — they are the
   runner's honesty mechanism, not noise:
   - `boundary` = an **expected failure** whose negative result is archived. Currently **2 boundary
     stages**: `RL·M6.2 structure-editing boundary` (0/20 robust failure) and
     `theory·cross-domain transfer·family-level` (T2 capacity not extrapolable).
   - `known-issue` = a defect that is **visible and tracked but not disguised as passing**. It is a
     first-class state: the registry measurement-vs-truth check is a **regular gate** that guards the whole
     class.
   - A gate result is read as `PASS / boundary / known-issue / FAIL` — a run with boundaries is not a
     clean run, and saying so is mandatory.
3. **Always carry the scope declaration** with any claim. The project's own required form is e.g.: the
   verifier is self-built; external corroboration covers 5/27 families; T2 absolute capacity does not
   pass; real free-text requirements are untested; RL makes no performance claim; the data is not
   experimental authority; "the B-parser domain-generalisation result is not a claim about the whole
   framework".
4. **Cite by protocol, not by number alone.** The same metric appears under different corpora and
   conventions (LOOCV **87.0%** @169 anchors vs **88.1%** @159 anchors + production switch; LODO transductive
   **85.2%** vs **84.9%**). State the corpus and the convention with every number.
5. The underlying reasoning, the diagnoses and the complete record live in the **internal development log
   (not shipped)**.

---

## 7. Repository and release state

**Privacy scan.** `tools/privacy_scan.py` applies **13 pre-registered rules** plus credential-filename rules
and a whitelist, and is **wired into the release-copy flow**: the copy is scanned automatically after being
made, and any hit exits non-zero. Current run: **0 hits — the release copy is publishable**. The scan finds no
API keys / tokens / passwords / private keys (`sk-`, `AKIA`, `ghp_`, `AIza`, `xox`, `BEGIN PRIVATE KEY`, JWT),
no credential-style filenames, no emails / phone numbers / IPs / social handles, **no personal names**
(DB signature fields are `"reviewer":"reviewer+user"` and `"reviewers":["user"]`; LICENSE says
`contributors`), no paths or usernames inside the `.pt` weights, and no absolute paths in code.

**Release copy.** Built by `tools/make_release_copy.py` (explicit inventory, dry-run by default, **copy-only,
never move**, refuses a non-empty target, **mirror prune** guarded by three safety gates: only inside the
target; the target must look like this project's copy; every item printed) into a
dedicated release copy directory outside the working tree.
It carries **only the reader-facing documents** — `README.md` / `README_cn.md`,
`CLAIMS_AND_BOUNDARIES.md` / `_cn.md`, `CREDIBILITY.md`, `cross domain rl architecture plan.md`, `LICENSE`,
`requirements.txt` — plus the code, the five databases, the model weights and the summary evidence CSVs;
the internal working material (full Chinese development log, full Chinese plan, release checklist, DB1
review artefacts) stays in the development tree and is **not shipped**. Shipped release set:
**1280 files / 28.9 MB**; the development tree it was built from (internal log, review artefacts,
regenerated outputs) is larger and **not shipped**. Regenerated outputs (reflow jsonl files, dataset
snapshots, audit logs, per-run training metrics) are **permanently excluded** from version control by
`.gitignore` **and** by the copy exclusions.

**Third-party material.** Third-party book/paper notes are **not part of the repository** (copyright);
**8 ambiguous items** (suspected textbook summaries) are left in place for the maintainer's own copyright
judgement, and the release action itself (`git init` / `git add -A` / commit) is left to the maintainer.

**Model weights shipped in-repo.** `b_network/bv1_model.pt` (**0.33 MB**) and
`b_network/bv1_model_cal.pt` (**0.31 MB**) — **0.64 MB total** — are published with the repository, so a
fresh clone has the full parser; the large A-side weights remain excluded. `README.md` documents the
retrain/recalibrate commands (`bv1_train_eval.py` → `bv1_model.pt`; `bv1_calibrate.py` →
`bv1_model_cal.pt`), the missing-weight behaviour (no error, graceful v0 fallback:
`status ∈ {auto, human_queue, v0_fallback}`) and the τ note (production is explicitly **τ = 0.50**).

**Bilingual documentation.** The published set carries parallel English/Chinese versions under one convention
— **the original filename holds the English version, the same name plus `_cn` holds the Chinese version**:
`README.md` / `README_cn.md` and `CLAIMS_AND_BOUNDARIES.md` / `CLAIMS_AND_BOUNDARIES_cn.md`, each with an
`English ｜ 中文` switcher line at the top. `CREDIBILITY.md` and `cross domain rl architecture plan.md` are
**English-only in the published set**; their full Chinese counterparts stay internal.

**Confidence calibration (shipped).** **Platt** scaling: ECE **0.050** (uncalibrated **0.066**); coverage at
95% precision **88% / 87%** (uncalibrated / Platt); `a = 2.03`, `b = −0.06`. Temperature scaling (fold
T = 1.53–4.02, mean 2.42) is **not** used: 9 validation samples cannot fit a temperature.

---

*Every figure in this file is the value to cite; the protocol column states the scale and convention it belongs
to.*
