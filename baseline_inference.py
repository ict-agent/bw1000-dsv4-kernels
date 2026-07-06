"""Baseline inference performance measurement."""
import requests
import time
import json

URL = "http://127.0.0.1:30001/v1/completions"

def measure():
    # Warmup
    for _ in range(3):
        requests.post(URL, json={"model":"default","prompt":"Hello","max_tokens":1,"temperature":0})

    print("=" * 60)
    print("BASELINE INFERENCE PERFORMANCE")
    print("=" * 60)

    # Prefill throughput
    print("\n--- Prefill (input ~4000 tokens, output 1) ---")
    prompt = "The quick brown fox jumps over the lazy dog. " * 400
    t0 = time.time()
    r = requests.post(URL, json={"model":"default","prompt":prompt,"max_tokens":1,"temperature":0})
    t1 = time.time()
    usage = r.json()["usage"]
    pt = usage["prompt_tokens"]
    print("  Input tokens: %d" % pt)
    print("  Time: %.3fs" % (t1-t0))
    print("  Prefill throughput: %.0f tokens/s" % (pt/(t1-t0)))

    # Decode throughput
    print("\n--- Decode (input short, output 128) ---")
    t0 = time.time()
    r = requests.post(URL, json={"model":"default","prompt":"Once upon a time","max_tokens":128,"temperature":0.5})
    t1 = time.time()
    usage = r.json()["usage"]
    ot = usage["completion_tokens"]
    print("  Output tokens: %d" % ot)
    print("  Time: %.3fs" % (t1-t0))
    print("  Decode throughput: %.1f tokens/s" % (ot/(t1-t0)))
    text = r.json()["choices"][0]["text"]
    print("  Text: %s..." % text[:80])

    # Concurrent requests
    print("\n--- Concurrent (8 requests, each 32 tokens) ---")
    import concurrent.futures
    def single_req(i):
        t0 = time.time()
        r = requests.post(URL, json={"model":"default","prompt":"Count from %d: " % i,"max_tokens":32,"temperature":0})
        return time.time() - t0, r.json()["usage"]["completion_tokens"]

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(single_req, i) for i in range(8)]
        results = [f.result() for f in futures]
    total_time = time.time() - t0
    total_tokens = sum(r[1] for r in results)
    print("  Total tokens: %d" % total_tokens)
    print("  Wall time: %.3fs" % total_time)
    print("  Aggregate throughput: %.1f tokens/s" % (total_tokens/total_time))

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    measure()
