"""Tests for T1-13's author-bench fixtures (#3068)."""

import importlib

from scripts.author_bench_fixtures import HUMAN_LABELED_SPOT_CHECK, TASKS


def _resolve(ref: str):
    module_name, qualname = ref.split(":")
    return getattr(importlib.import_module(module_name), qualname)


class TestTasks:
    def test_has_ten_tasks(self) -> None:
        assert len(TASKS) == 10

    def test_task_ids_are_unique(self) -> None:
        ids = [t.task_id for t in TASKS]
        assert len(ids) == len(set(ids))

    def test_every_expected_sequence_entry_resolves_to_a_real_callable(self) -> None:
        for task in TASKS:
            assert len(task.expected_sequence) >= 2
            for ref in task.expected_sequence:
                assert callable(_resolve(ref))

    def test_nl_prompts_are_non_empty(self) -> None:
        for task in TASKS:
            assert task.nl_prompt.strip()


class TestHumanLabeledSpotCheck:
    def test_has_at_least_thirty_items(self) -> None:
        assert len(HUMAN_LABELED_SPOT_CHECK) >= 30

    def test_every_item_references_a_real_task(self) -> None:
        task_ids = {t.task_id for t in TASKS}
        for item in HUMAN_LABELED_SPOT_CHECK:
            assert item.task_id in task_ids

    def test_expected_outputs_are_independently_reproducible(self) -> None:
        """Second, independent check on the hand computation: re-derive every
        expected_output by actually executing the real callable chain, rather than
        trusting the hand-verified value at face value.
        """
        tasks_by_id = {t.task_id: t for t in TASKS}
        for item in HUMAN_LABELED_SPOT_CHECK:
            task = tasks_by_id[item.task_id]
            value = item.sample_input
            for ref in task.expected_sequence:
                value = _resolve(ref)(value)
            assert value == item.expected_output, (
                f"{item.task_id} @ x={item.sample_input}: hand-computed "
                f"{item.expected_output}, but executing {task.expected_sequence} gives {value}"
            )

    def test_covers_every_task_at_least_once(self) -> None:
        covered = {item.task_id for item in HUMAN_LABELED_SPOT_CHECK}
        assert covered == {t.task_id for t in TASKS}
