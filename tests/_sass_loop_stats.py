"""
SASS dependency analysis for one kernel of the built .so: for every LDSM (ldmatrix) find the distance
(in instructions) to the first HMMA that consumes one of its destination registers. If most
distances are tiny the fragment loads are not hoisted and their latency is exposed to the tensor
pipe. Also prints the instruction mix. Usage:
  python tests/_sass_loop_stats.py <path-to-.so> <mangled-name-regex>
"""
import re, subprocess, sys, collections
so, pat = sys.argv[1], sys.argv[2]
txt = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True).stdout
funcs = re.split(r"\n\s*Function : ", txt)
for f in funcs[1:]:
    name, body = f.split("\n", 1)
    if not re.search(pat, name):
        continue
    ins = []
    for line in body.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;", line)
        if m:
            ins.append(m.group(1))
    mix = collections.Counter(i.split()[0].split(".")[0].lstrip("@!P0123456 ") for i in ins)
    # opcode extraction robust to predication
    def opc(i):
        t = i.split()
        if t and t[0].startswith("@"):
            t = t[1:]
        return t[0].split(".")[0] if t else "?"
    mix = collections.Counter(opc(i) for i in ins)
    print(f"== {name[:110]}")
    print("   total", len(ins), " ".join(f"{k}={v}" for k, v in mix.most_common(14)))
    dists = []
    for idx, i in enumerate(ins):
        if opc(i) != "LDSM":
            continue
        m = re.search(r"LDSM\S*\s+R(\d+)", i)
        if not m:
            continue
        base = int(m.group(1)); n = 4 if ".4" in i.split()[0] or ".x4" in i else 2
        dst = {f"R{base + k}" for k in range(n)}
        for j in range(idx + 1, min(idx + 400, len(ins))):
            if opc(ins[j]) == "HMMA":
                srcs = set(re.findall(r"R\d+", ins[j]))
                if dst & srcs:
                    dists.append(j - idx); break
    if dists:
        h = collections.Counter(min(d, 40) // 4 * 4 for d in dists)
        print("   LDSM->dependent HMMA distance histogram (instr, bucket of 4, 40+ capped):", " ".join(f"[{k}-{k+3}]:{v}" for k, v in sorted(h.items())))
        print(f"   median distance {sorted(dists)[len(dists)//2]} over {len(dists)} LDSM; HMMA/LDSM = {mix['HMMA']/max(1,mix['LDSM']):.2f}")
