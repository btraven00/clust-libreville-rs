#!/usr/bin/env python3
"""prof.py PROFILE.json.gz [top]  -- self time by innermost (inlined) function and by source line,
from `samply record -s --unstable-presymbolicate -r 4000 -o PROFILE.json.gz -- cmd` (needs
perf_event_paranoid <= 1). Reads the .json.syms.json sidecar next to the profile.
from a samply recording made with --save-only --unstable-presymbolicate."""
import bisect, gzip, json, sys
from collections import Counter

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
p = json.load(gzip.open(path))
syms = json.load(open(path.replace(".json.gz", ".json.syms.json")))
st = syms["string_table"]
tables = {}
for lib in syms["data"]:
    t = sorted(lib["symbol_table"], key=lambda e: e["rva"])
    tables[lib["debug_name"]] = ([e["rva"] for e in t], t)

def resolve(libname, rva):
    if libname not in tables:
        return (libname, "?", 0)
    rvas, t = tables[libname]
    i = bisect.bisect_right(rvas, rva) - 1
    if i < 0 or rva >= t[i]["rva"] + t[i]["size"]:
        return (libname, "?", 0)
    e = t[i]
    fr = e.get("frames") or []
    if fr:
        f = fr[-1]  # innermost inline frame
        return (st[f["function"]], st[f.get("file", 0)] if f.get("file") is not None else "", f.get("line", 0))
    return (st[e["symbol"]], "", 0)

by_func, by_line, total = Counter(), Counter(), 0
for th in p["threads"]:
    s, stk, fr, res = th["samples"], th["stackTable"], th["frameTable"], th["resourceTable"]
    w = s.get("weight") or [1] * s["length"]
    cache = {}
    for i in range(s["length"]):
        sid = s["stack"][i]
        if sid is None:
            continue
        f = stk["frame"][sid]
        if f not in cache:
            r = th["funcTable"]["resource"][fr["func"][f]]
            lib = p["libs"][res["lib"][r]]["debugName"] if r is not None and r >= 0 and res["lib"][r] is not None else "?"
            cache[f] = resolve(lib, fr["address"][f])
        fn, file, line = cache[f]
        n = w[i] or 1
        total += n
        by_func[fn] += n
        by_line[(fn, file.split("/")[-1], line)] += n

print(f"{path}: {total} samples (all threads)")
for fn, n in by_func.most_common(top):
    print(f"  {100 * n / total:5.1f}%  {fn[:110]}")
print("  -- by line --")
for (fn, file, line), n in by_line.most_common(top):
    print(f"  {100 * n / total:5.1f}%  {file}:{line}  {fn[:70]}")
