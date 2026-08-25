# Kernel-Optimization Grounding Report — NotebookLM "kernel optimization" corpus

Date: 2026-08-25 | Branch: feat/profiling-stage2-evidence
Method: 12 NotebookLM (`nlm`) queries over the 25-source corpus (queries q02–q05 inherited from a prior worker; q06–q12 run this session). Full answers live in `.praxia/research/nlm_kernelopt/q*.json`; only ≤3000-char extracts were read during synthesis. Citations are bracketed source names; no page numbers are asserted because nlm answers did not expose stable page anchors.

Purpose: ground the `xtrax-optimizing` skill (scope: `.praxia/docs/specs/260825_jax-optimizing-skill-scope.md`) in the literature. Our taxonomy: **T1** host-boundary mechanics, **T2** data movement, **T3** composition. Our discipline: ProbeRecord claim-gated measurement (STRUCTURAL / DISPATCH_COUNT / TERM_RANKING / END_TO_END), fail-closed keeping, paired configs.

