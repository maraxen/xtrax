# P1.5 topo manifest artifacts

Versioned manifests produced by recon (P1.5-TOPO). **Artifact only — not a gate** in v0.1.

Resolved via `port/port_target.toml` → `port/manifests/<wave_id>.toml`.

## Schema (AC-11)

```toml
[manifest]
wave_id = "wave_001_example"
task_id = "260617_xtrax-composition-mission"
manifest_hash = "sha256:<sha256-of-canonical-toml-bytes>"
created_at = "2026-06-18T00:00:00Z"
paper_mask_enforced = true

[[kernels]]
order = 1
qualname = "xtrax.transforms.map.apply_map"
module_path = "src/xtrax/transforms/map.py"
depends_on = []
```

- **`manifest_hash`**: SHA-256 of canonical TOML bytes (not `content_hash` of individual kernels).
- **`task_id`**: Port wave task id; stale manifests (hash mismatch or older `task_id`) WARN at P2 entry, FAIL at Phase 2 integration entry.
