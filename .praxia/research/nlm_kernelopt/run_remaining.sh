#!/bin/bash
cd /home/marielle/projects/xtrax
DIR=.praxia/research/nlm_kernelopt
LOG=$DIR/run_q06_q12.log
: > "$LOG"
run_one() {
  local name="$1"; shift
  local q="$1"
  if [ -s "$DIR/$name.json" ] && grep -q '"answer"' "$DIR/$name.json"; then echo "SKIP $name exists" >>"$LOG"; return 0; fi
  local ok=0 rc
  for attempt in 1 2; do
    echo "RUN $name attempt=$attempt $(date -Is)" >>"$LOG"
    timeout 180 nlm notebook query kernelopt "$q" > "$DIR/$name.json" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q '"answer"' "$DIR/$name.json"; then
      ok=1; echo "OK $name rc=0" >>"$LOG"; break
    fi
    echo "FAIL $name attempt=$attempt rc=$rc size=$(stat -c%s "$DIR/$name.json" 2>/dev/null || echo 0)" >>"$LOG"
    if [ $attempt -eq 1 ]; then sleep 20; fi
  done
  [ $ok -eq 0 ] && { echo "GIVEUP $name after 2 attempts" >>"$LOG"; mv "$DIR/$name.json" "$DIR/$name.failed.txt" 2>/dev/null; }
  sleep 5
}
run_one q06_mixed_precision_scaling "What mixed-precision guidance exists (FP8 formats paper, Micikevicius mixed precision): when is reduced precision safe, what scaling/granularity strategies exist?"
run_one q07_cuda_best_practices_occupancy "What do the CUDA Best Practices Guide and Hopper whitepaper say about occupancy vs other bottlenecks, async memcpy, warp primitives, and when kernel fusion helps?"
run_one q08_gpudirect_pagedattention "What data-movement optimizations do GPUDirect Storage and PagedAttention describe?"
run_one q09_jax11_primitives "From the JAX 0.11 reference PDF: what primitive-level features matter for performance work (donate_argnums semantics, remat/checkpointing, shard_map, pallas/CustomCall)?"
run_one q10_bio_workload_kernels "How do AlphaFold, MMseqs2, ADEPT structure compute kernels for sequence/biology workloads - tiling strategies, memory access patterns, custom CUDA use?"
run_one q11_measurement_methodology "Which papers report measured speedups with methodology details worth imitating (paired baselines, variance handling, ablations)?"
run_one q12_contradictions "What tensions/contradictions appear across sources (fusion enthusiasm vs separation-of-concerns; autotuning vs hand-tuning)?"
echo "ALLDONE $(date -Is)" >>"$LOG"
