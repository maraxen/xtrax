"""Minimal pytest fixture that always fails — empirical-oracle self-test repro."""


def test_audit_repro_always_fails() -> None:
    assert False, "empirical-oracle self-test repro"
