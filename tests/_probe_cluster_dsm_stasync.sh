#!/usr/bin/env bash
# ptxas feature probe for the sm90-style 2-CTA cluster DSM crossover on PLAIN sm_120:
#   A: st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64  (authors' peer store)
#   B: mapa.shared::cluster.u32                                            (peer addr, replaces ^16777216)
#   C: mbarrier.arrive.shared::cluster.b64 (remote arrive)                 (fallback handshake leg)
#   D: st.shared::cluster.v2.s64 (plain peer store)                        (fallback data leg)
#   E: mbarrier.arrive.expect_tx.shared::cta.b64                           (tx-barrier init leg)
# Each probed standalone so one rejection cannot mask another.
set -u
NVCC=/usr/local/cuda-13.0/bin/nvcc
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

probe() {
    local name="$1" body="$2"
    cat > "$TMP/p.cu" <<EOF
#include <cstdint>
__global__ void k(uint64_t* g, uint32_t sa, uint32_t bar, int64_t x, int64_t y) {
    $body
}
EOF
    if "$NVCC" -std=c++17 -arch=sm_120 -c "$TMP/p.cu" -o "$TMP/p.o" 2> "$TMP/err.txt"; then
        echo "[ACCEPT] $name"
    else
        echo "[REJECT] $name :: $(grep -m1 -oE "(ptxas.*|error.*)" "$TMP/err.txt" | head -1)"
    fi
}

probe "A st.async.weak.shared::cluster.mbarrier v2.s64" \
'asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3];" :: "r"(sa), "l"(x), "l"(y), "r"(bar));'

probe "B mapa.shared::cluster.u32" \
'uint32_t peer; asm volatile("mapa.shared::cluster.u32 %0, %1, 1;" : "=r"(peer) : "r"(sa)); g[0] = peer;'

probe "C mbarrier.arrive remote (shared::cluster)" \
'asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(bar));'

probe "D plain st.shared::cluster.v2.s64" \
'asm volatile("st.shared::cluster.v2.s64 [%0], {%1, %2};" :: "r"(sa), "l"(x), "l"(y));'

probe "E mbarrier.arrive.expect_tx.shared::cta.b64" \
'uint64_t st; asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], 36864;" : "=l"(st) : "r"(bar));'

echo "DSM_PROBE_DONE"
