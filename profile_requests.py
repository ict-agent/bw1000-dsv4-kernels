"""Profile-aware request driver for the sglang server.

Sends prefill-heavy requests (large input, short output) so the HIP wrappers
actually run (prefill is NOT in the cuda graph, so python wrappers execute and
SGLANG_HIP_PROFILE timing is recorded). Also sends decode requests for e2e timing
(these go through graph replay; wrappers are not invoked, so they don't add
profiling entries — but e2e latency is captured for A/B).

After requests, dumps profiling.json (periodic flush happens every 256 wrapper
calls automatically; this also reads it back).
"""
import requests, time, json, os, sys

HOST = os.environ.get("HOST_IP", "127.0.0.1")
PORT = os.environ.get("PORT", "30001")
URL = f"http://{HOST}:{PORT}/generate"
TAG = os.environ.get("MODE_TAG", "prof_winners")

def wait_ready(timeout=2400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://{HOST}:{PORT}/get_model_info", timeout=5)
            if r.status_code == 200:
                print(f"[ready] server up after {time.time()-t0:.0f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(10)
    print("[ready] TIMEOUT", flush=True)
    return False

def req(input_len, output_len):
    prompt = "word " * (input_len // 2)
    t0 = time.time()
    try:
        r = requests.post(URL, json={"text": prompt,
            "sampling_params": {"max_new_tokens": output_len, "temperature": 0}}, timeout=300)
        dt = (time.time() - t0) * 1000
        r.json()
        return dt
    except Exception as e:
        print(f"  req err: {e}", flush=True)
        return None

def bench(label, cases, n=5):
    print(f"=== {label} ===", flush=True)
    out = []
    for il, ol in cases:
        dts = [d for d in (req(il, ol) for _ in range(n)) if d]
        dts.sort()
        med = dts[len(dts)//2] if dts else 0
        print(f"  in={il:>5} out={ol:>4}: median={med:.1f}ms (n={len(dts)})", flush=True)
        out.append({"in": il, "out": ol, "median_ms": round(med,1)})
    return out

if __name__ == "__main__":
    if not wait_ready():
        sys.exit(1)
    # warmup (also flushes any capture-time state)
    req(128, 4)
    # PREFILL-heavy: large input, short output -> wrappers execute (not in graph)
    # keep input_len <= 8192: 16384/32768 OOM-crash TP workers at mem_frac=0.76
    prefill = bench("prefill-heavy (wrappers run)", [(2048, 16), (4096, 16), (8192, 16)], n=5)
    # DECODE-heavy: small input, long output -> graph replay (wrappers NOT invoked)
    decode = bench("decode-heavy (graph replay)", [(128, 256), (512, 256), (4096, 256)], n=5)
    # let periodic flush land
    time.sleep(3)
    json.dump({"prefill": prefill, "decode": decode},
              open(f"/workspace/hip_kernels/results/reqbench_{TAG}.json","w"), indent=2)
    print(f"=== profiling.json ===", flush=True)
    try:
        print(open("/workspace/hip_kernels/results/profiling.json").read(), flush=True)
    except Exception as e:
        print(f"  read err: {e}", flush=True)
    print("DONE", flush=True)
