"""Parity tests verifying bit-for-bit equivalence with aminx parameter hashing (GATE-11).

Covering [T5-01], GATE-11, and AC-02 from SPEC-261105-CAMPAIGN-RUNNER:
- Bit-for-bit equivalence between compute_unit_input_hash, ManifestRow.input_hash,
  and production aminx golden vectors.
- Negative control: perturb float from 0.1 to 0.1000001 and verify hash flip.
- Negative controls: key mutations, seed divergence, policy mutations.
- Zero-float normalization parity (-0.0 to 0.0) across Python and NumPy primitives.
- Invariance under dict key order changes.
- Exclusion of CAMPAIGN_OWNED_KEYS from parameter sub-dictionary hashing.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from xtrax.campaign.manifest import (
    CAMPAIGN_OWNED_KEYS,
    build_manifest_row,
    compute_unit_input_hash,
)

# Golden test vectors pre-computed from production aminx campaigns
GOLDEN_VECTORS: list[dict[str, Any]] = [
    {
        "name": "ligandmpnn_standard_chunk0",
        "inputs": {
            "campaign_id": "c_synth_01",
            "job_id": "job_000",
            "chunk_id": 0,
            "sample_start": 0,
            "sample_count": 100,
            "target_input": "targets/synth_01.pdb",
            "arm_id": "arm_01",
            "arm_backend": "boltz2",
            "checkpoint_id": "ligandmpnn_v_32_020_25",
            "fixed_policy": "none",
            "state_weight_profile": "equal",
            "seed": 42,
            "parameters": {"temperature": 0.1, "use_ligand": True},
        },
        "expected_input_hash": "67e12b7f7ff84d2fc14515e5b230a391fb2fb739c6003a952d26d649e38a704b",
    },
    {
        "name": "ligandmpnn_standard_chunk1_offset",
        "inputs": {
            "campaign_id": "c_synth_01",
            "job_id": "job_000",
            "chunk_id": 1,
            "sample_start": 100,
            "sample_count": 100,
            "target_input": "targets/synth_01.pdb",
            "arm_id": "arm_01",
            "arm_backend": "boltz2",
            "checkpoint_id": "ligandmpnn_v_32_020_25",
            "fixed_policy": "none",
            "state_weight_profile": "equal",
            "seed": 42,
            "parameters": {"temperature": 0.1, "use_ligand": True},
        },
        "expected_input_hash": "008707e599c87443702a5b9423f8fdc5915409afebeae6a2d793fbbb83d321a9",
    },
    {
        "name": "proteinmpnn_zero_bias_normalized",
        "inputs": {
            "campaign_id": "c_synth_01",
            "job_id": "job_001",
            "chunk_id": 0,
            "sample_start": 0,
            "sample_count": 50,
            "target_input": "targets/prot_bias.pdb",
            "arm_id": "arm_02",
            "arm_backend": "chai1",
            "checkpoint_id": "proteinmpnn_v_48_020",
            "fixed_policy": "none",
            "state_weight_profile": "equal",
            "seed": 100,
            "parameters": {"bias": 0.0, "threshold": 0.0},
        },
        "expected_input_hash": "f37f61f6cab040582d3881acd6970116a5d36df88cc1a96e8bf6ce9ef27e2aee",
    },
    {
        "name": "proteinmpnn_ca_only_interface_heavy",
        "inputs": {
            "campaign_id": "c_aminx_prod_02",
            "job_id": "job_prod_01",
            "chunk_id": 2,
            "sample_start": 200,
            "sample_count": 100,
            "target_input": "targets/complex_prod.pdb",
            "arm_id": "arm_prod",
            "arm_backend": "boltz2",
            "checkpoint_id": "proteinmpnn_v_48_020",
            "fixed_policy": "ca_only",
            "state_weight_profile": "interface_heavy",
            "seed": 12345,
            "parameters": {"model_weights": "original", "sampling_temp": 0.2},
        },
        "expected_input_hash": "da9775b6bf008892ac2c6e0c198d89edf1bebcff087b833843237cd7948e92e5",
    },
    {
        "name": "complex_slice_nested_arrays",
        "inputs": {
            "campaign_id": "c_complex_slice",
            "job_id": "job_slice_00",
            "chunk_id": 0,
            "sample_start": 0,
            "sample_count": 250,
            "target_input": "targets/multimer.pdb",
            "arm_id": "arm_tied",
            "arm_backend": "protenix",
            "checkpoint_id": "model_v_48_020",
            "fixed_policy": "tied_multimer",
            "state_weight_profile": "asymmetric",
            "seed": 999,
            "parameters": {
                "bias_by_res": [0.5, 1.0, 0.0],
                "num_clashes": 0,
                "use_mask": True,
            },
        },
        "expected_input_hash": "421eb1c0336e39dce8735ee2428ecaea7637a23d8ad497d09db184d7ba781275",
    },
]


def test_aminx_manifest_slice_parity() -> None:
    """Verify bit-for-bit parity with pre-computed aminx golden vectors across all fields."""
    for vec in GOLDEN_VECTORS:
        row = build_manifest_row(**vec["inputs"])
        assert (
            row.input_hash == vec["expected_input_hash"]
        ), f"Row input_hash diverged on vector {vec['name']}: {row.input_hash} != {vec['expected_input_hash']}"

        computed = compute_unit_input_hash(
            spec_dict=vec["inputs"]["parameters"],
            campaign_id=vec["inputs"]["campaign_id"],
            job_id=vec["inputs"]["job_id"],
            chunk_id=vec["inputs"]["chunk_id"],
            sample_start=vec["inputs"]["sample_start"],
            sample_count=vec["inputs"]["sample_count"],
            checkpoint_id=vec["inputs"]["checkpoint_id"],
            fixed_policy=vec["inputs"]["fixed_policy"],
            state_weight_profile=vec["inputs"]["state_weight_profile"],
            seed=vec["inputs"]["seed"],
        )
        assert (
            computed == vec["expected_input_hash"]
        ), f"compute_unit_input_hash diverged on vector {vec['name']}: {computed} != {vec['expected_input_hash']}"

        # Manifest row hash must be a valid 64-char hex string
        assert len(row.manifest_row_hash) == 64
        assert int(row.manifest_row_hash, 16) > 0


def test_gate_11_negative_control_float_perturbation() -> None:
    """GATE-11 Parity Gate & AC-02 Negative Control: Float perturbation causes hash flip.

    Modifying temperature from 0.1 to 0.1000001 must produce a divergent input_hash.
    """
    baseline_vec = GOLDEN_VECTORS[0]
    baseline_inputs = baseline_vec["inputs"]

    # 1. Confirm baseline matches golden vector
    baseline_hash = compute_unit_input_hash(
        spec_dict=baseline_inputs["parameters"],
        campaign_id=baseline_inputs["campaign_id"],
        job_id=baseline_inputs["job_id"],
        chunk_id=baseline_inputs["chunk_id"],
        sample_start=baseline_inputs["sample_start"],
        sample_count=baseline_inputs["sample_count"],
        checkpoint_id=baseline_inputs["checkpoint_id"],
        fixed_policy=baseline_inputs["fixed_policy"],
        state_weight_profile=baseline_inputs["state_weight_profile"],
        seed=baseline_inputs["seed"],
    )
    assert baseline_hash == baseline_vec["expected_input_hash"]

    # 2. Perturb float: 0.1 -> 0.1000001
    perturbed_params = dict(baseline_inputs["parameters"])
    perturbed_params["temperature"] = 0.1000001

    perturbed_hash = compute_unit_input_hash(
        spec_dict=perturbed_params,
        campaign_id=baseline_inputs["campaign_id"],
        job_id=baseline_inputs["job_id"],
        chunk_id=baseline_inputs["chunk_id"],
        sample_start=baseline_inputs["sample_start"],
        sample_count=baseline_inputs["sample_count"],
        checkpoint_id=baseline_inputs["checkpoint_id"],
        fixed_policy=baseline_inputs["fixed_policy"],
        state_weight_profile=baseline_inputs["state_weight_profile"],
        seed=baseline_inputs["seed"],
    )

    assert perturbed_hash != baseline_hash, (
        f"GATE-11 VIOLATION: Perturbing float from 0.1 to 0.1000001 did not change input_hash! "
        f"Hash remained {baseline_hash}"
    )

    # 3. Test through ManifestRow construction as well
    row_perturbed_inputs = copy.deepcopy(baseline_inputs)
    row_perturbed_inputs["parameters"]["temperature"] = 0.1000001
    row_perturbed = build_manifest_row(**row_perturbed_inputs)
    assert row_perturbed.input_hash == perturbed_hash
    assert row_perturbed.input_hash != baseline_vec["expected_input_hash"]


def test_zero_float_normalization_parity() -> None:
    """Verify that negative zero (-0.0) strictly normalizes to 0.0 with 0 bit divergence."""
    # Test with standard python float -0.0
    hash_pos = compute_unit_input_hash(
        spec_dict={"bias": 0.0, "threshold": 0.0},
        campaign_id="c_synth_01",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=50,
        checkpoint_id="proteinmpnn_v_48_020",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=100,
    )

    hash_neg = compute_unit_input_hash(
        spec_dict={"bias": -0.0, "threshold": -0.0},
        campaign_id="c_synth_01",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=50,
        checkpoint_id="proteinmpnn_v_48_020",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=100,
    )
    assert hash_pos == hash_neg
    assert hash_pos == "f37f61f6cab040582d3881acd6970116a5d36df88cc1a96e8bf6ce9ef27e2aee"

    # Test with NumPy floating types
    hash_np = compute_unit_input_hash(
        spec_dict={"bias": np.float32(-0.0), "threshold": np.float64(-0.0)},
        campaign_id="c_synth_01",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=50,
        checkpoint_id="proteinmpnn_v_48_020",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=100,
    )
    assert hash_np == hash_pos


def test_dict_key_ordering_invariance() -> None:
    """Verify that parameter dict key insertion ordering does not affect input_hash."""
    dict_order_1 = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}
    dict_order_2 = {"gamma": 3.0, "alpha": 1.0, "beta": 2.0}
    dict_order_3 = {"beta": 2.0, "gamma": 3.0, "alpha": 1.0}

    common_kwargs = {
        "campaign_id": "c_order_test",
        "job_id": "job_01",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 10,
        "checkpoint_id": "chk_01",
        "fixed_policy": "none",
        "state_weight_profile": "equal",
        "seed": 42,
    }

    h1 = compute_unit_input_hash(spec_dict=dict_order_1, **common_kwargs)
    h2 = compute_unit_input_hash(spec_dict=dict_order_2, **common_kwargs)
    h3 = compute_unit_input_hash(spec_dict=dict_order_3, **common_kwargs)

    assert h1 == h2 == h3


def test_campaign_owned_keys_exclusion() -> None:
    """Verify that CAMPAIGN_OWNED_KEYS present in spec_dict are stripped and do not alter hash."""
    clean_params = {"temperature": 0.1, "use_ligand": True}
    tainted_params = dict(clean_params)
    for key in CAMPAIGN_OWNED_KEYS:
        tainted_params[key] = f"spurious_{key}_value"

    common_kwargs = {
        "campaign_id": "c_synth_01",
        "job_id": "job_000",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 100,
        "checkpoint_id": "ligandmpnn_v_32_020_25",
        "fixed_policy": "none",
        "state_weight_profile": "equal",
        "seed": 42,
    }

    h_clean = compute_unit_input_hash(spec_dict=clean_params, **common_kwargs)
    h_tainted = compute_unit_input_hash(spec_dict=tainted_params, **common_kwargs)

    assert h_clean == h_tainted
    assert h_clean == "67e12b7f7ff84d2fc14515e5b230a391fb2fb739c6003a952d26d649e38a704b"


@pytest.mark.parametrize(
    ("field_name", "perturbed_value"),
    [
        ("seed", 43),
        ("fixed_policy", "ca_only"),
        ("state_weight_profile", "interface_heavy"),
        ("checkpoint_id", "ligandmpnn_v_32_010_10"),
        ("campaign_id", "c_synth_02"),
        ("job_id", "job_001"),
        ("chunk_id", 5),
        ("sample_start", 50),
        ("sample_count", 200),
    ],
)
def test_discrete_parameter_perturbations(field_name: str, perturbed_value: Any) -> None:
    """Verify that any mutation to top-level execution parameters flips the input_hash."""
    base_kwargs: dict[str, Any] = {
        "campaign_id": "c_synth_01",
        "job_id": "job_000",
        "chunk_id": 0,
        "sample_start": 0,
        "sample_count": 100,
        "checkpoint_id": "ligandmpnn_v_32_020_25",
        "fixed_policy": "none",
        "state_weight_profile": "equal",
        "seed": 42,
    }
    spec_dict = {"temperature": 0.1, "use_ligand": True}

    baseline_hash = compute_unit_input_hash(spec_dict=spec_dict, **base_kwargs)

    perturbed_kwargs = dict(base_kwargs)
    perturbed_kwargs[field_name] = perturbed_value

    perturbed_hash = compute_unit_input_hash(spec_dict=spec_dict, **perturbed_kwargs)
    assert perturbed_hash != baseline_hash, f"Mutating {field_name} did not flip input_hash"
