## (g) Source coverage table

Identification method: `nlm` exposes source IDs (UUIDs), not titles; identities below were inferred from the leading text of each source's first cited chunk (`references[].cited_text`) across all 12 answers. Inferences marked ⚠ are lower-confidence. "Q" = queries whose answer drew on that source.

| Source (inferred identity) | ID | Cited by | Status |
|---|---|---|---|
| FlashAttention (Dao et al., arXiv 2205.14135) | 9d4ae950 | q02 q04 q11 q12 | core |
| FlashAttention-3 (Shah et al.) | 3ed43e6e | q02 q03 q04 q06 q07 q11 q12 | core |
| Dissecting the NVIDIA Hopper Architecture (µbench) | 4aef3fe9 | q03 q04 q07 q12 | good |
| Microbenchmarking study w/ T4/P4 (⚠ Turing-era) | 3c0e6e5e | q04 q12 | thin |
| Second microbenchmarking study (⚠ Volta-era; "bare-metal peak inaccessible to plain CUDA") | 22ab01a2 | q12 | thin |
| Triton (Tillet et al.) | 8c69fe39 | q05 q11 q12 | good |
| Halide (Ragan-Kelley et al.) | 57c99885 | q04 q05 q11 q12 | good |
| Ansor (Zheng et al., arXiv 2006.06762) | bebb11b3 | q05 q11 q12 | good |
| MARLIN | 96ae5bec | q04 q07 q11 | good |
| PagedAttention / vLLM | 5365100b | q08 q11 | good |
| FP8 formats paper (NVIDIA/Arm/Intel) | c761f479 | q06 | focused |
| Mixed-Precision Training (Micikevicius et al.) | 31b6314f | q06 | focused |
| CUDA C++ Best Practices Guide | c6acfa61 | q07 | single-query |
| CUDA C++ Programming Guide (r13.3) | 31f4cc99 | q03 q04 q07 q12 | good |
| GPUDirect Storage Overview Guide | db840e88 | q08 | single-query |
| Nsight Compute Kernel Profiling Guide | 6a6bb659 | q04 q07 | good |
| NVIDIA H100 whitepaper | ba7760bd | q03 q04 q07 | good |
| AlphaFold main paper (Jumper et al.) | 6575255c | q10 | thin |
| AlphaFold SI (memory-consumption section) | ca78c2d6 | q10 | thin |
| MMseqs2 | dddfbc83 | q10 | thin |
| ADEPT | 07f3c182 | q10 q11 | good |
| JAX 0.11 reference PDF | ae6d8565 | q03 q04 q09 q12 | core |
| Probable-but-unconfirmed: classical Roofline paper (Williams et al.) | cb57ee01 ⚠ | q04 q06 | uncertain |

**Never cited by any of the 12 answers:** **Foldcomp** — no query targeted it and it never appeared in `sources_used`. If its content matters (compression of structure ensembles is adjacent to our staging/sink concerns), it needs a dedicated follow-up query.

**Coverage caveats (honesty ledger):**
1. The notebook reportedly holds **25 sources; only 21 distinct IDs** appeared across all answers. Beyond Foldcomp, up to three uploads were either never retrieved or are duplicates/splits (e.g., PagedAttention and vLLM may be separate uploads; only one was cited).
2. The classical **Roofline paper's own text was never distinctly attributed**: q04's ridge-point/ceilings content is backed citation-wise by the Nsight Compute guide's roofline-chapter plus microbenchmarking sources [Nsight Compute], [Hopper µbench]. One candidate ID (cb57ee01) fits but is unconfirmed. Treat "roofline facts" above as *profiler-operationalized* roofline, not primary-source scholarship.
3. **Questions this corpus cannot answer for us:** (i) direct literature on *host-callback* boundary mechanics (our T1) — the corpus is device-centric; T1 grounding rests on analogy (producer/consumer decoupling [FA-3]; bounce-buffer elimination [GPUDirect Storage]); (ii) deep `pallas`/`CustomCall` API mechanics — q09 surfaced only the opacity trade-off, not usage detail; (iii) anything about Foldcomp.

