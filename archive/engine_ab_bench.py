"""Engine A/B benchmark: TTFT/TPOT/throughput.
Config A = original SGLang W8A8 (slimquant_marlin + lmslim Triton + lightop C++)
Config B = SGLang W8A8 with HIP bit-exact quant kernel patched in
Run this script against each config; pass label as argv[1].
"""
import requests, time, json, sys, concurrent.futures

URL="http://127.0.0.1:30001"
LABEL=sys.argv[1] if len(sys.argv)>1 else "config"

def ttft_tpot(prompt, max_tokens, tag):
    t0=time.time(); first=None; toks=0; times=[]
    try:
        r=requests.post(URL+"/v1/completions",json={"model":"default","prompt":prompt,"max_tokens":max_tokens,"temperature":0,"stream":True},stream=True,timeout=180)
        for line in r.iter_lines():
            if line:
                s=line.decode()
                if s.startswith("data: ") and s[6:]!="[DONE]":
                    try: json.loads(s[6:]); now=time.time()
                    except: continue
                    if first is None: first=now
                    times.append(now); toks+=1
    except Exception as e: return {"tag":tag,"error":str(e)[:60]}
    te=time.time(); ttft=(first-t0)*1000 if first else 0
    its=[(times[i]-times[i-1])*1000 for i in range(1,len(times))] if len(times)>1 else [0]
    tpot=sum(its)/len(its) if its else 0
    p99=sorted(its)[min(int(len(its)*.99),len(its)-1)] if its else 0
    return {"tag":tag,"ttft_ms":round(ttft,1),"mean_tpot_ms":round(tpot,1),"p99_tpot_ms":round(p99,1),"tokens":toks,"total_ms":round((te-t0)*1000,1)}

def main():
    R={"label":LABEL,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),"tests":[]}
    for _ in range(3): requests.post(URL+"/v1/completions",json={"model":"default","prompt":"w","max_tokens":1,"temperature":0},timeout=60)
    print("=== %s ==="%LABEL)
    # decode 32
    r=ttft_tpot("Tell me a story about a robot learning to paint:",32,"decode32"); print("  decode32: TTFT=%.1f TPOT=%.1f p99=%.1f"%(r["ttft_ms"],r["mean_tpot_ms"],r["p99_tpot_ms"])); R["tests"].append(r)
    # decode 64
    r=ttft_tpot("Write a short essay on the future of AI:",64,"decode64"); print("  decode64: TTFT=%.1f TPOT=%.1f"%(r["ttft_ms"],r["mean_tpot_ms"])); R["tests"].append(r)
    # prefill 1k
    r=ttft_tpot("Explain quantum computing. "*50,16,"prefill1k_decode16"); print("  1k_prefill: TTFT=%.1f"%(r["ttft_ms"])); R["tests"].append(r)
    # prefill 4k
    r=ttft_tpot("The quick brown fox. "*400,1,"prefill4k"); print("  4k_prefill: TTFT=%.1f (%.0f tok/s)"%(r["ttft_ms"],4000/(r["ttft_ms"]/1000) if r["ttft_ms"] else 0)); R["tests"].append(r)
    # concurrent 8
    def one(i):
        t0=time.time(); requests.post(URL+"/v1/completions",json={"model":"default","prompt":"Count %d"%i,"max_tokens":16,"temperature":0},timeout=60); return time.time()-t0
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: [f.result() for f in [ex.submit(one,i) for i in range(8)]]
    wall=time.time()-t0; R["tests"].append({"tag":"concurrent8x16","wall_s":round(wall,3),"agg_tok_s":round(128/wall,1)}); print("  concurrent8: wall=%.2fs agg=%.1f tok/s"%(wall,128/wall))
    with open("/workspace/engine_ab_%s.json"%LABEL,"w") as f: json.dump(R,f,indent=2)
    print("Saved: /workspace/engine_ab_%s.json"%LABEL)

if __name__=="__main__": main()
