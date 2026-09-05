#!/usr/bin/env bash
# ptxas feature probe (compile-only) for the open items from the sm100 recon:
#   packed fp32 math (fma/add/mul.f32x2), cp.async.bulk.prefetch.L2, stmatrix, setmaxnreg
# on PLAIN sm_120 under the cu13 ptxas. ACCEPT = assembles; REJECT = ptxas error.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
PTXAS="$CUDA_HOME/bin/ptxas"
OUT=/tmp/_sm120_extra_probes; mkdir -p "$OUT"
emit() {
    local name="$1"; shift
    cat > "$OUT/$name.ptx" <<PTX
.version 9.0
.target sm_120
.address_size 64
.visible .global .align 16 .f32 gbuf[64];
.visible .entry probe_$name()
{
    .reg .f32 %f<8>;
    .reg .b64 %fd<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .shared .align 16 .b8 sbuf[1024];
    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f40000000;
    mov.u64 %rd1, gbuf;
    mov.u32 %r1, sbuf;
    $*
    ret;
}
PTX
}
emit fma_f32x2      'mov.b64 %fd1, {%f1, %f2}; mov.b64 %fd2, {%f2, %f1}; fma.rn.f32x2 %fd3, %fd1, %fd2, %fd1;'
emit add_f32x2      'mov.b64 %fd1, {%f1, %f2}; add.rn.f32x2 %fd3, %fd1, %fd1;'
emit mul_f32x2      'mov.b64 %fd1, {%f1, %f2}; mul.rn.f32x2 %fd3, %fd1, %fd1;'
emit bulk_prefetch  'cp.async.bulk.prefetch.L2.global [%rd1], 256;'
emit stmatrix_x4    'stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r1], {%r2, %r3, %r4, %r5};'
emit stmatrix_x4_t  'stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r1], {%r2, %r3, %r4, %r5};'
emit setmaxnreg     'setmaxnreg.inc.sync.aligned.u32 232;'
emit ex2_f32x2      'mov.b64 %fd1, {%f1, %f2}; ex2.approx.ftz.f32x2 %fd3, %fd1;'
emit cp_bulk_g2s    'mov.u32 %r2, sbuf; cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r2], [%rd1], 256, [%r2];'
for f in "$OUT"/*.ptx; do
    n="$(basename "$f" .ptx)"
    if "$PTXAS" -arch=sm_120 -o /dev/null "$f" 2>"$OUT/$n.err"; then echo "[ACCEPT] $n"
    else echo "[REJECT] $n :: $(head -1 "$OUT/$n.err" | cut -c1-110)"; fi
done
echo "EXTRA_PROBE_DONE"
# --- second batch (sm90 recon verify-list item 3) ---
emit cpasync_mbar_noinc 'mov.u32 %r2, sbuf; cp.async.mbarrier.arrive.noinc.shared.b64 [%r2];'
emit cpasync_l2hint     'mov.u32 %r2, sbuf; createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0; cp.async.cg.shared.global.L2::cache_hint [%r2], [%rd1], 16, %rd2;'
emit cpasync_l2_256b    'mov.u32 %r2, sbuf; cp.async.cg.shared.global.L2::256B [%r2], [%rd1], 16;'
emit createpolicy_first 'createpolicy.fractional.L2::evict_first.b64 %rd2, 1.0;'
emit griddep_wait       'griddepcontrol.wait;'
emit griddep_launch     'griddepcontrol.launch_dependents;'
emit ld_nc_l1noalloc    'ld.global.nc.L1::no_allocate.f32 %f3, [%rd1];'
emit cvt_e4m3x2_f16x2   '.reg .b16 %h<2>; mov.b16 %h1, 0x3838; cvt.rn.f16x2.e4m3x2 %r3, %h1;'
for n in cpasync_mbar_noinc cpasync_l2hint cpasync_l2_256b createpolicy_first griddep_wait griddep_launch ld_nc_l1noalloc cvt_e4m3x2_f16x2; do
    f="$OUT/$n.ptx"
    if "$PTXAS" -arch=sm_120 -o /dev/null "$f" 2>"$OUT/$n.err"; then echo "[ACCEPT] $n"
    else echo "[REJECT] $n :: $(head -1 "$OUT/$n.err" | cut -c1-110)"; fi
done
echo "EXTRA_PROBE2_DONE"
