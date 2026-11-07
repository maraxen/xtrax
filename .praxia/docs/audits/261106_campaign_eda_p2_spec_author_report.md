---
category: audit
title: 'Handoff Report: Campaign EDA Formal Technical Specification (Phase P2)'
description: Formal technical specification author handoff delivering initial r1.0
  spec, superseding 261028 Decision 1, mapping Q1-Q5 to data sources, and defining
  DomainPlugin protocol
task_id: 261106_xtrax-campaign-eda
status: complete
date: '2026-11-06'
agent: teamwork_worker_spec_1
verdict: PASS
target_specification: .praxia/docs/specs/261106_campaign-eda-redesign.md
---

# Handoff Report: Campaign EDA Formal Technical Specification (Phase P2)

- **Agent**: `teamwork_worker_spec_1`
- **Role**: Campaign EDA Specification Author
- **Working Directory**: `/home/marielle/projects/xtrax/.agents/teamwork/teamwork_worker_spec_1`
- **Parent Conversation ID**: `8c6c9b8f-26e4-41f2-9b35-adf51fb4ef0f`
- **Target Specification Artifact**: `/home/marielle/projects/xtrax/.praxia/docs/specs/261106_campaign-eda-redesign.md`
- **Milestone Status**: Phase P2 Specification PASSED & COMPLETE

---

## 1. Observation

1. **Original Request & Requirements**:
   - `ORIGINAL_REQUEST.md:5-91`: Mandates building Campaign EDA for `xtrax` to evaluate what knob settings produced across campaigns, answering five scientific questions (library composition, scores, knob effects with honest uncertainty, coverage, and secondary run telemetry) via typed domain plugins (`aminx`, `demistify`), closing debts #3301, #3143, #3300, and `demistify` #830.
   - Enforces strict boundaries: core `xtrax` depends strictly on `stdlib + numpy`; `duckdb`, `pyarrow`, and `altair` reside strictly behind optional extras; zero framework bridges (`maraxiom`, `bathos`, `cisternal`) inside `xtrax.eda`; zero cross-imports between `xtrax.eda` and `maraxiom.eda`.
2. **Phase P0 Reconnaissance Synthesis**:
   - `P0_RECON_SYNTHESIS.md:9-99`: Established the Master Question $\to$ Data Source $\to$ Field Mapping Table; grounded in `origin/feat/261105-campaign-runner-track5` (`manifest.py:126-363`, `marker.py:87-363`, `lease.py:57`), `origin/main` (`src/xtrax/run/unit_index.py:741-755`), `xtrax.pack` (`src/xtrax/pack/_node.py:506-672`), and prototype findings (`campaign_library_eda_prototype_v2.py:140-348`).
3. **Phase P1 Research Note**:
   - `261106_campaign_eda_prior_art_research.md:1-527`: Master Verdict Registry adopting full factorial Cartesian sweep grids, vectorized cluster bootstrap ($W \cdot Y / U$), MDE power discipline ($\text{MDE}_{0.80} \approx 2.802 \cdot \text{SE}$), small-sample guards ($U < 3 \implies \text{descriptive\_only}$), designable masking, alphabet normalization, and a dual-layer visual presentation (static hashed SVG reviewed object + progressive airgapped HTML with 2-second timeout guard).
4. **Draft Respec & Prior Spec State**:
   - `.praxia/docs/specs/261028_xtrax-eda-redesign.md:45-50`: Decision 1 originally deferred campaign-level tables until downstream workloads articulated concrete schema requirements.
   - `docs/261106-campaign-eda-respec:.praxia/docs/specs/261106_campaign-library-eda-respec.md:1-250`: Documented prototype findings on real TEV design AR campaign data (`6d38ebaf`), proving that padding tokens inflate entropy ($H \approx 4.04$ bits), alphabet mismatches drop recovery (0.14 vs 0.45), and uninstrumented pack telemetry requires explicit bannering.

---

## 2. Logic Chain

1. **Need for Formal Specification**:
   - From Observation 1, the task mandates delivering a complete technical specification for Phase P2 before entering Phase P3 Adversarial Review.
2. **Superseding Decision 1**:
   - From Observation 4, Decision 1 of `261028_xtrax-eda-redesign.md` explicitly deferred campaign-level tables pending concrete downstream needs.
   - From Observations 1 and 2, `aminx` (generative sequences) and `demistify` (molecular dynamics ensembles) now require campaign EDA, fulfilling the condition. Therefore, Decision 1 is formally superseded, elevating campaign tables to active, first-class status while preserving compiler plan inspection as the secondary telemetry view (§1.1).
3. **Architectural Isolation (Decision 3 & F-10)**:
   - From Observation 1 and 3, maintaining zero external framework bridges and zero cross-imports between `xtrax.eda` and `maraxiom.eda` requires that core `xtrax` depend strictly on `stdlib + numpy`.
   - All analytical SQL and graphic generation dependencies (`duckdb`, `pyarrow`, `altair`, `vl-convert`) are partitioned into optional extras `xtrax[eda-campaign]` and `xtrax[eda-viz]`, with neutral file artifacts (JSON, Parquet, Vega-Lite, SVG) serving as the universal integration seam (§1.2, §1.3, §6.1).
4. **Storage Format Parity & Unit Index**:
   - From Observation 2, `xtrax.pack.read_unit_member()` provides random-access frame decompression for `.tar.zst` archives with byte-identical parity to loose directories.
   - Grounding the EDA reader strictly on `unit_index.lookup()` keyed by `manifest_row_hash` guarantees that only verified, committed units are pooled, while uncommitted, partial, or quarantined units are refused and bannered (§3.3, §3.4).
5. **Domain Plugin Quarantine**:
   - From Observations 1 and 2, domain-specific extractors (`aminx`, `demistify`) must not destabilize the core engine. Standardizing on the `xtrax.eda.domains` entry-point group with `copy.deepcopy()` input isolation and exception quarantine ensures robust, fault-tolerant execution (§4.1, §4.2).
6. **Vectorized Statistical Rigor**:
   - From Observation 3, generative and simulation regimes suffer from pseudoreplication if sequences/frames are treated as independent degrees of freedom ($N \to \infty$).
   - Grounding uncertainty in the global `md-replicas-are-samples` rule, executing vectorized cluster bootstrap ($W \cdot Y / U$), enforcing small-sample guards ($U < 3$), and calculating Minimum Detectable Effect ($\text{MDE}_{0.80}$) provides honest, publication-grade statistical inference (§5.3).
7. **Execution Partitioning**:
   - From Observations 1 and 2, 10 modular, file-disjoint fixer tasks are specified to enable clean parallel implementation in Phase P5 (§9).

---

## 3. Caveats

- **Upstream Campaign Runner Merge**: `origin/feat/261105-campaign-runner-track5` is currently unmerged to `main`. When runner PR #267 merges to `main`, the implementation branch should be rebased with `git rebase --onto origin/main origin/feat/261105-campaign-runner-track5`.
- **Debt #3300 Runtime Seam**: Execution loop pack event telemetry remains uninstrumented at runtime until debt #3300 is completed. As specified, the Secondary Run View explicitly banners `"not instrumented (debt #3300)"` while reporting static archive status (`is_packed`).
- **Consumer Sink Writer Updates**: While `aminx` and `demistify` readers can parse existing stores and fall back to PDB parsing, full production fidelity for target residue tables requires consumer sink writers to embed canonical `ResidueTable` records at write time.

---

## 4. Conclusion

The formal technical specification for `xtrax` Campaign EDA has been authored and committed to:
`/home/marielle/projects/xtrax/.praxia/docs/specs/261106_campaign-eda-redesign.md`

The specification comprehensively addresses all requirements from `ORIGINAL_REQUEST.md` and the orchestrator dispatch:
- Supersedes Decision 1 and reaffirms Decision 3 and F-10 boundaries.
- Specifies complete data models (`CampaignManifest`, `ManifestRow`, `manifest_row_hash`, `SinkSchema`, `LibraryCell`, `ResidueTable`).
- Defines the `xtrax.eda.domains` plugin protocol with deep-copy isolation and quarantine.
- Specifies mathematical formulations for all 5 scientific questions with full BLAS/NumPy vectorization.
- Defines neutral export bundles and the dual-layer Reviewed Object visual architecture.
- Outlines the 5 verification controls, the end-to-end ubiquitin aminx demo campaign, and 10 discrete fixer tasks for Phase P5.

The task is complete and ready for Phase P3 Adversarial Review.

---

## 5. Verification Method

1. **Verify File Existence & Frontmatter**:
   ```bash
   head -n 20 /home/marielle/projects/xtrax/.praxia/docs/specs/261106_campaign-eda-redesign.md
   ```
   *Expected*: Valid YAML frontmatter with `task_id: "261106_xtrax-campaign-eda"` and `status: "FINAL r1.0"`.
2. **Verify Section Coverage**:
   Inspect the file to confirm all 10 major sections are present:
   - §1: Architectural Governance & Decision Records (Decision 1 superseded, Decision 3 & F-10 restated)
   - §2: Master Analytical Mapping (5 Scientific Questions Table)
   - §3: Data Model, Schemas & Storage Parity (ManifestRow, Knob Taxonomy, Unit Index, Loose vs Packed)
   - §4: Typed Domain Plugin Architecture (Protocol, Quarantine, aminx, demistify)
   - §5: Vectorized Analytical Engine (Q1-Q5 mathematical formulations and algorithms)
   - §6: Neutral Export Artifacts & Visualization Architecture (Parquet, JSON, Vega-Lite, Static SVG, Airgapped HTML)
   - §7: Verification & Statistical Controls Suite (5 Controls)
   - §8: Real End-to-End aminx Demo Campaign (Ubiquitin 1ubq)
   - §9: Discrete Fixer Tasks for Phase P5 Implementation (10 Work Packages)
   - §10: Acceptance Criteria & Audit Checklist (16 Criteria AC-01 to AC-16)
3. **Invalidation Conditions**:
   - The specification is invalidated if any internal imports of `maraxiom`, `bathos`, or `cisternal` are introduced into `xtrax.eda`.
   - The specification is invalidated if python loops over sample/draw axes are reintroduced into the statistical engine.
