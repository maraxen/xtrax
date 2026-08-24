# tests/profiling/fixtures/

`records/*.json` -- committed ProbeRecord fixtures for `test_report.py`,
copied verbatim from prolix
`tests/profiling/fixtures/records/` (branch wt-20260807-132628) on
2026-08-24. Generated only via `ProbeRecord.write` (never hand-authored
JSON). Round-trip is asserted in
`test_every_committed_fixture_roundtrips_probe_record_read`. The scope names
inside them (pme_*, flash_*, dense_*) are prolix MD vocabulary -- they are
opaque strings as far as this contract is concerned and exist to exercise
the report renderer's sorting/banner/absent-rendering behavior. When xtrax
grows its own first probes (Phase B of
`.praxia/docs/specs/260824_upstream-profiling-probe-tooling-from-prolix.md`),
add xtrax-native fixtures beside these rather than editing them.

NOT ported (yet): prolix's `b1_water_stage1.trace.json.gz` real-trace
fixture and its companion test leg. That leg regenerates matching compiled
HLO text by re-calling prolix domain scripts (`profile_b1_water_trace.py`),
so it cannot run here until an xtrax-native program produces an equivalent
(trace, HLO-text) fixture pair. Until then the parser's ground-truth check
rests on the synthetic legs in `test_trace_parse.py`; the prolix suite keeps
covering the real leg end to end.
