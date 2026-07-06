"""Benchmark sglang server: decode + prefill latency at various sizes."""
import requests, time, json, sys, os
HOST = os.environ.get("HOST_IP", "127.0.0.1")
PORT = os.environ.get("PORT", "30001")
URL = f"http://{HOST}:{PORT}/generate"
TAG = os.environ.get("MODE_TAG", "run")

def bench(input_len, output_len, n=20, warmup=3):
    # build a prompt of ~input_len tokens
    prompt = "word " * (input_len // 2)
    for _ in range(warmup):
        requests.post(URL, json={"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0}})
    ts = []
    for _ in range(n):
        t0 = time.time()
        r = requests.post(URL, json={"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0}})
        ts.append(time.time() - t0)
        try:
            r.json()
        except Exception:
            pass
    ts.sort()
    med = ts[len(ts)//2]
    # e2e latency; decode TTOT = (e2e - prefill) / output_len approx
    print(f"  in={input_len:>5} out={output_len:>4}: median={med*1000:.1f}ms  ttot~={(med*1000 - 50)/max(output_len,1):.2f}ms/tok  (n={n})")
    return {"in": input_len, "out": output_len, "median_ms": round(med*1000,1)}

print(f"=== Benchmark {TAG} ===")
R = []
print("decode (small in, long out):")
for il, ol in [(128, 256), (512, 256), (4096, 256)]:
    R.append(bench(il, ol))
print("prefill (large in, short out):")
for il, ol in [(4096, 16), (16384, 16), (32768, 16)]:
    R.append(bench(il, ol))
out = f"/workspace/hip_kernels/results/bench_{TAG}.json"
json.dump(R, open(out, "w"), indent=2)
print(f"Saved {out}")
