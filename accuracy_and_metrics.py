"""Accuracy probe + full TTFT/TPOT/throughput metrics on real model.
6 GSM8K-style math questions with deterministic decoding (temp=0).
Since our HIP kernel is bit-exact, accuracy MUST be identical to baseline.
We verify by checking outputs are sensible (correct math answers).
"""
import requests, time, json, concurrent.futures

URL="http://127.0.0.1:30001"

QUESTIONS=[
    ("What is 7 * 8? Answer with just the number.", "56"),
    ("What is 144 / 12? Answer with just the number.", "12"),
    ("What is 25 + 37? Answer with just the number.", "62"),
    ("What is 9 * 13? Answer with just the number.", "117"),
    ("What is 100 - 45? Answer with just the number.", "55"),
    ("What is 2^10? Answer with just the number.", "1024"),
]

def accuracy_probe():
    print("="*60); print("ACCURACY PROBE (6 math questions, temp=0)"); print("="*60)
    correct=0
    for i,(q,expected) in enumerate(QUESTIONS):
        r=requests.post(URL+"/v1/completions",json={"model":"default","prompt":q,"max_tokens":16,"temperature":0},timeout=120)
        txt=r.json()["choices"][0]["text"].strip()
        ok = expected in txt
        correct += ok
        print("  Q%d: %s -> '%s' [%s]"%(i+1,q[:40],txt[:40],"✓" if ok else "✗ expected="+expected))
    print("  Score: %d/%d"%(correct,len(QUESTIONS)))
    return {"correct":correct,"total":len(QUESTIONS)}

def metrics():
    print("\n"+"="*60); print("TTFT/TPOT/THROUGHPUT"); print("="*60)
    # warmup
    for _ in range(3): requests.post(URL+"/v1/completions",json={"model":"default","prompt":"warmup","max_tokens":1,"temperature":0},timeout=60)
    results={}

    # decode TTFT/TPOT
    prompt="Tell me a story about a robot:"
    t0=time.time(); first=None; toks=0; times=[]
    for line in requests.post(URL+"/v1/completions",json={"model":"default","prompt":prompt,"max_tokens":32,"temperature":0,"stream":True},stream=True,timeout=180).iter_lines():
        if line:
            s=line.decode()
            if s.startswith("data: ") and s[6:]!="[DONE]":
                try: json.loads(s[6:]); now=time.time()
                except: continue
                if first is None: first=now
                times.append(now); toks+=1
    tend=time.time()
    ttft=(first-t0)*1000 if first else 0
    its=[(times[i]-times[i-1])*1000 for i in range(1,len(times))] if len(times)>1 else [0]
    tpot=sum(its)/len(its) if its else 0
    p99=sorted(its)[min(int(len(its)*.99),len(its)-1)] if its else 0
    results["decode32"]={"ttft_ms":round(ttft,1),"mean_tpot_ms":round(tpot,1),"p99_tpot_ms":round(p99,1),"decode_tok_s":round(toks/(tend-t0),2) if tend-t0>0 else 0}
    print("  decode32: TTFT=%.1fms TPOT=%.1fms p99=%.1fms throughput=%.1f tok/s"%(ttft,tpot,p99,results["decode32"]["decode_tok_s"]))

    # prefill 4k
    prompt="The quick brown fox. "*400
    t0=time.time()
    requests.post(URL+"/v1/completions",json={"model":"default","prompt":prompt,"max_tokens":1,"temperature":0},timeout=120)
    tend=time.time()
    ttft=(tend-t0)*1000
    results["prefill_4k"]={"ttft_ms":round(ttft,1),"prefill_tok_s":round(4000/(tend-t0),0)}
    print("  prefill_4k: TTFT=%.1fms prefill=%.0f tok/s"%(ttft,4000/(tend-t0)))

    # concurrent 8
    def one(i):
        t0=time.time()
        requests.post(URL+"/v1/completions",json={"model":"default","prompt":"Count %d"%i,"max_tokens":16,"temperature":0},timeout=60)
        return time.time()-t0
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        [f.result() for f in [ex.submit(one,i) for i in range(8)]]
    wall=time.time()-t0
    results["concurrent8"]={"wall_s":round(wall,3),"agg_tok_s":round(128/wall,1)}
    print("  concurrent8: wall=%.2fs agg=%.1f tok/s"%(wall,128/wall))
    return results

def main():
    R={"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
    R["accuracy"]=accuracy_probe()
    R["metrics"]=metrics()
    with open("/workspace/accuracy_and_metrics.json","w") as f: json.dump(R,f,indent=2)
    print("\nSaved: /workspace/accuracy_and_metrics.json")

if __name__=="__main__": main()
