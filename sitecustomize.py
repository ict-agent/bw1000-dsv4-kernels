"""Auto-loaded by Python at startup (via PYTHONPATH). Imports HIP DSV4 integration
so monkey-patches apply as sglang modules load. Gated by SGLANG_USE_HIP_DSV4=1."""
import os
try:
    if os.environ.get("SGLANG_USE_HIP_DSV4") == "1":
        import hip_dsv4_integration  # noqa: F401
except Exception:
    pass
