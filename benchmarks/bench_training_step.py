def test_trainer_step_throughput(benchmark, trainer, trainer_state, synthetic_batch):
    # Declaration protocol for XTRAX_BENCH_RECORD_DIR emission (see
    # xtrax.profiling.bench).
    benchmark.extra_info.update(
        {
            "xtrax_stage": 1,
            "xtrax_n_atoms": 32,
            "xtrax_scale_basis": "batch_rows",
        }
    )
    state = trainer_state
    for _ in range(3):  # JIT warmup
        state, _ = trainer.step(state, synthetic_batch)

    def one_step():
        return trainer.step(state, synthetic_batch)

    benchmark.pedantic(one_step, rounds=5, iterations=20)
