"""Parity tests verifying bit-for-bit equivalence with aminx parameter hashing."""

from xtrax.campaign.manifest import (
    build_manifest_row,
    compute_unit_input_hash,
)

GOLDEN_VECTORS = [
    {
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
    }
]


def test_aminx_manifest_slice_parity() -> None:
    for vec in GOLDEN_VECTORS:
        row = build_manifest_row(**vec["inputs"])
        assert row.input_hash == vec["expected_input_hash"]

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
        assert computed == vec["expected_input_hash"]
