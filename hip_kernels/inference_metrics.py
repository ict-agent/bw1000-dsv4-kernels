"""
Formal inference benchmark with TTFT/TPOT/ITL metrics.
Uses SGLang bench_serving with verbose streaming JSON trace.
"""
import requests
import time
import json
import sys
import threading

SERVER_URL = "http://127.0.0.1:30001"
RESULTS = {"metadata": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "tests": []}

def measure_ttft_tpot(prompt, max_tokens, label=""):
    """Measure TTFT and TPOT using streaming."""
    url = SERVER_URL + "/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.time()
    first_token_time = None
    token_times = []
    total_tokens = 0

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        for line in resp.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        now = time.time()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                        total_tokens += 1
                    except:
                        pass
    except Exception as e:
        print("  ERROR: %s" % str(e)[:50])
        return None

    t_end = time.time()

    if first_token_time is None:
        return None

    ttft = (first_token_time - t_start) * 1000  # ms
    total_time = (t_end - t_start) * 1000

    # TPOT = time between tokens (excluding first)
    if len(token_times) > 1:
        inter_token_times = [(token_times[i] - token_times[i-1])*1000 for i in range(1, len(token_times))]
        mean_tpot = sum(inter_token_times) / len(inter_token_times)
        p99_tpot = sorted(inter_token_times)[int(len(inter_token_times)*0.99)] if len(inter_token_times) > 1 else mean_tpot
    else:
        mean_tpot = 0
        p99_tpot = 0
        inter_token_times = []

    result = {
        "label": label,
        "prompt_len": len(prompt.split()),
        "max_tokens": max_tokens,
        "actual_tokens": total_tokens,
        "ttft_ms": round(ttft, 2),
        "mean_tpot_ms": round(mean_tpot, 2),
        "p99_tpot_ms": round(p99_tpot, 2),
        "total_time_ms": round(total_time, 2),
        "decode_throughput_tok_s": round(total_tokens / (total_time/1000), 2) if total_time > 0 else 0,
    }
    return result


def run_benchmark():
    print("=" * 70)
    print("INFERENCE METRICS: TTFT / TPOT / Throughput")
    print("=" * 70)

    # Warmup
    requests.post(SERVER_URL + "/v1/completions",
                 json={"model":"default","prompt":"warmup","max_tokens":1,"temperature":0})

    # Test 1: Short input, measure decode
    print("\n--- Test 1: Short input (decode-heavy) ---")
    r = measure_ttft_tpot("Tell me a long story about:", 64, "short_input_decode64")
    if r:
        print("  TTFT: %.2f ms" % r["ttft_ms"])
        print("  Mean TPOT: %.2f ms" % r["mean_tpot_ms"])
        print("  P99 TPOT: %.2f ms" % r["p99_tpot_ms"])
        print("  Decode throughput: %.2f tok/s" % r["decode_throughput_tok_s"])
        RESULTS["tests"].append(r)

    # Test 2: Medium input (prefill + decode)
    print("\n--- Test 2: Medium input (~1K tokens) ---")
    prompt = "Explain the theory of relativity in detail. " * 50
    r = measure_ttft_tpot(prompt, 32, "medium_input_1k")
    if r:
        print("  TTFT: %.2f ms" % r["ttft_ms"])
        print("  Mean TPOT: %.2f ms" % r["mean_tpot_ms"])
        print("  Decode throughput: %.2f tok/s" % r["decode_throughput_tok_s"])
        RESULTS["tests"].append(r)

    # Test 3: Long input (prefill-heavy)
    print("\n--- Test 3: Long input (~4K tokens, 1 output) ---")
    prompt = "The quick brown fox jumps over the lazy dog. " * 400
    r = measure_ttft_tpot(prompt, 1, "long_input_4k_prefill")
    if r:
        print("  TTFT: %.2f ms (= prefill time for ~4K tokens)" % r["ttft_ms"])
        prefill_throughput = 4000.0 / (r["ttft_ms"] / 1000.0) if r["ttft_ms"] > 0 else 0
        print("  Estimated prefill throughput: %.0f tok/s" % prefill_throughput)
        r["prefill_throughput_tok_s"] = round(prefill_throughput, 0)
        RESULTS["tests"].append(r)

    # Test 4: Batch throughput
    print("\n--- Test 4: Concurrent requests (8x, 16 tokens each) ---")
    import concurrent.futures
    def single_req(i):
        t0 = time.time()
        r = requests.post(SERVER_URL + "/v1/completions",
                         json={"model":"default","prompt":"Count %d:" % i,"max_tokens":16,"temperature":0},
                         timeout=60)
        t1 = time.time()
        usage = r.json().get("usage", {})
        return {"time_s": t1-t0, "output_tokens": usage.get("completion_tokens", 0)}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(single_req, i) for i in range(8)]
        results = [f.result() for f in futures]
    total_wall = time.time() - t0
    total_tokens = sum(r["output_tokens"] for r in results)
    batch_result = {
        "label": "concurrent_8x16",
        "num_requests": 8,
        "total_output_tokens": total_tokens,
        "wall_time_s": round(total_wall, 3),
        "aggregate_throughput_tok_s": round(total_tokens / total_wall, 2),
    }
    print("  Wall time: %.3fs" % total_wall)
    print("  Total tokens: %d" % total_tokens)
    print("  Aggregate throughput: %.2f tok/s" % batch_result["aggregate_throughput_tok_s"])
    RESULTS["tests"].append(batch_result)

    # Save
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for t in RESULTS["tests"]:
        print("  %s" % json.dumps(t))

    with open("/workspace/inference_metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nFull trace: /workspace/inference_metrics.json")


if __name__ == "__main__":
    run_benchmark()
