"""Sharper s1: forward jaxpr holds the intent primitive, not a scan; control: the
reference holds a scan. Imports the spike module (its prints run first)."""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spike_loop_primitive as s  # noqa: E402

fwd = str(jax.make_jaxpr(s.f_loop)(s.x))
ref = str(jax.make_jaxpr(s.f_ref)(s.x))
print("s1b forward jaxpr: xtrax_loop", "xtrax_loop" in fwd, "| scan", "scan[" in fwd)
print("s1b control ref jaxpr: scan", "scan[" in ref)
