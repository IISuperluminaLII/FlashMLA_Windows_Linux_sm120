"""One-shot patch: make tests/bench_bwd_vs_sdpa.py's batch/heads env-overridable (BENCH_B/BENCH_H)."""
import io, re, sys
p = "tests/bench_bwd_vs_sdpa.py"
s = io.open(p, encoding="utf-8").read()
if "BENCH_B" in s:
    print("[OK] already patched"); sys.exit(0)
anchor = "PEAK = 251.9          # RTX PRO 6000 Blackwell dense bf16 TFLOPS"
assert anchor in s, "PEAK anchor missing"
s = s.replace(anchor, anchor + "\nBENCH_B = int(os.environ.get(\"BENCH_B\", \"2\")); BENCH_H = int(os.environ.get(\"BENCH_H\", \"8\"))   # shape overrides", 1)
s2, n = re.subn(r"run\(2,\s*S,\s*8", "run(BENCH_B, S, BENCH_H", s)
print("[OK] run() call sites patched:", n)
io.open(p, "w", encoding="utf-8", newline="\n").write(s2)
