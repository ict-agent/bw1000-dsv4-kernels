"""Compare HIP-on vs HIP-off e2e results (A/B). Reads reqbench_*.json."""
import json, sys, os

def load(name):
    try:
        return {f"{r['in']}/{r['out']}": r["median_ms"] for r in json.load(open(f"/workspace/hip_kernels/results/{name}")) if "median_ms" in r}
    except Exception:
        return {}

hip = {}
for tag in ["hipon_final2", "hipon_callsite2", "hipon_fixed_rope", "hipon_noprof"]:
    d = load(f"reqbench_{tag}.json")
    if d: hip = d; break

print(f"HIP-on source: {tag}")
print(f"{'case':<14}{'baseline':>12}{'HIP-on':>12}{'speedup':>10}")
for case in sorted(set(list(hip))):
    b = baseline.get(case); h = hip.get(case)
    if b and h:
        print(f"{case:<14}{b:>10.0f}ms{h:>10.0f}ms{h/b:>10.2f}x")
    elif h:
        print(f"{case:<14}{'?':>12}{h:>10.0f}ms{'?':>10}")
