def test_trainer_step_throughput(benchmark, trainer, trainer_state, synthetic_batch):
    state = trainer_state
    for _ in range(3):  # JIT warmup
        state, _ = trainer.step(state, synthetic_batch)

    def one_step():
        return trainer.step(state, synthetic_batch)

    benchmark.pedantic(one_step, rounds=5, iterations=20)
