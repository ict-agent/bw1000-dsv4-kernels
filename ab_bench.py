"""A/B benchmark: TTFT/TPOT/throughput for baseline vs patched.
Run twice (once per config), capture into JSON.
"""
import requests, time, json, sys, os

URL = "http://127.0.0.1:30001"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "unknown"

def bench_ttft_tpot(prompt, max_tokens, label):
    t_start = time.time()
    first_t = None
    toks = 0
    times = []
    try:
        r = requests.post(URL + "/v1/completions", json={"model":"default","prompt":prompt,"max_tokens":max_tokens,"temperature":0,"stream":True}, stream=True, timeout=180)
        for line in r.iter_lines():
            if line:
                s = line.decode()
                if s.startswith("data: "):
                    d = s[6:]
                    if d == "[DONE]": break
                    try:
                        json.loads(d)
                        now = time.time()
                        if first_t is None: first_t = now
                        times.append(now)
                        toks += 1
                    except: pass
    except Exception as e:
        return {"label":label,"error":str(e)[:80]}
    t_end = time.time()
    ttft = (first_t - t_start)*1000 if first_t else 0
    if len(times) > 1:
        its = [(times[i]-times[i-1])*1000 for i in range(1,len(times))]
        tpot = sum(its)/len(its)
        p99 = sorted(its)[min(int(len(its)*0.99), len(its)-1)]
    else:
        tpot = 0; p99 = 0
    return {"label":label,"ttft_ms":round(ttft,2),"mean_tpot_ms":round(tpot,2),
            "p99_tpot_ms":round(p99,2),"tokens":toks,"total_ms":round((t_end-t_start)*1000,2)}

def main():
    results = {"label":LABEL,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"tests":[]}
    # warmup
    for _ in range(3):
        requests.post(URL+"/v1/completions",json={"model":"default","prompt":"hi","max_tokens":1,"temperature":0},timeout=60)

    print("=== %s ===" % LABEL)
    # TTFT short
    r = bench_ttft_tpot("Tell me a story:", 32, "decode32_short")
    print("  decode32: TTFT=%.1fms TPOT=%.1fms p99=%.1fms" % (r["ttft_ms"], r.get("mean_tpot_ms",0), r.get("p99_tpot_ms",0)))
    results["tests"].append(r)
    # TTFT 1k
    r = bench_ttft_tpot("Explain relativity. "*50, 16, "decode16_1k")
    print("  decode16_1k: TTFT=%.1fms TPOT=%.1fms" % (r["ttft_ms"], r.get("mean_tpot_ms",0)))
    results["tests"].append(r)
    # Prefill 4k
    r = bench_ttft_tpot("The quick brown fox. "*400, 1, "prefill_4k")
    print("  prefill_4k: TTFT=%.1fms (=%.0f tok/s)" % (r["ttft_ms"], 4000/(r["ttft_ms"]/1000) if r["ttft_ms"] else 0))
    results["tests"].append(r)
    # Concurrent 8
    import concurrent.futures
    def one(i):
        t0=time.time()
        requests.post(URL+"/v1/completions",json={"model":"default","prompt":"Count %d"%i,"max_tokens":16,"temperature":0},timeout=60)
        return time.time()-t0
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        rs=[ex.submit(one,i) for i in range(8)]
        [f.result() for f in rs]
    wall=time.time()-t0
    print("  concurrent8: wall=%.2fs agg=%.1f tok/s" % (wall, 128/wall))
    results["tests"].append({"label":"concurrent8x16","wall_s":round(wall,3),"agg_tok_s":round(128/wall,2)})

    out = "/workspace/ab_%s.json" % LABEL
    with open(out,"w") as f: json.dump(results,f,indent=2)
    print("Saved: %s" % out)

if __name__ == "__main__":
    main()
