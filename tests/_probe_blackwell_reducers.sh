#!/usr/bin/env bash
# Probe which Blackwell-class REDUCER/ACCUMULATOR instructions ptxas accepts on
# sm_120 (plain) and sm_120a, under BOTH toolchains (12.9 = PTX 8.7, 13.0 = PTX 9.0).
# Same methodology as _probe_blockscale_targets.sh: one micro .ptx per instruction;
# ACCEPT = assembles, REJECT = ptxas error. Compile-only, no GPU touched.
set -u
OUT=/tmp/_bw_reducer_probes
mkdir -p "$OUT"

emit() {  # emit <name> <ptx-version> <body-lines...>
    local name="$1" ver="$2"; shift 2
    cat > "$OUT/$name.ptx" <<EOF
.version $ver
.target sm_120
.address_size 64
.visible .global .align 16 .f32 gbuf[8];
.visible .entry probe_$name()
{
    .reg .f32 %f<8>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .shared .align 16 .b8 sbuf[256];
    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f40000000;
    mov.u64 %rd1, gbuf;
    mov.u32 %r1, 0xffffffff;
    $*
    ret;
}
EOF
}

# --- candidates (PTX ISA syntax per NVIDIA docs) ---
emit redux_add_f32      8.7 'redux.sync.add.f32 %f3, %f1, %r1;'
emit redux_max_f32      8.7 'redux.sync.max.f32 %f3, %f1, %r1;'
emit redux_min_abs_f32  8.7 'redux.sync.min.abs.f32 %f3, %f1, %r1;'
emit red_v2_f32         8.7 'red.global.add.v2.f32 [%rd1], {%f1, %f2};'
emit red_v4_f32         8.7 'red.global.add.v4.f32 [%rd1], {%f1, %f2, %f1, %f2};'
emit atom_v2_f32        8.7 'atom.global.add.v2.f32 {%f3, %f4}, [%rd1], {%f1, %f2};'
emit atom_v4_f32        8.7 'atom.global.add.v4.f32 {%f3, %f4, %f5, %f6}, [%rd1], {%f1, %f2, %f1, %f2};'
emit red_bf16x2         8.7 '.reg .b32 %h<2>; mov.b32 %h1, 0x3f803f80; red.global.add.noftz.bf16x2 [%rd1], %h1;'
emit cp_reduce_bulk_f32 8.7 'mov.u32 %r2, sbuf; cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%rd1], [%r2], 128; cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;'

run_probe() {  # run_probe <ptxas-path> <label> <arch>
    local PTXAS="$1" label="$2" arch="$3"
    echo "=== $label arch=$arch ==="
    for f in "$OUT"/*.ptx; do
        local name; name="$(basename "$f" .ptx)"
        if "$PTXAS" -arch="$arch" -o /dev/null "$f" 2>"$OUT/$name.$label.$arch.err"; then
            echo "  [ACCEPT] $name"
        else
            echo "  [REJECT] $name :: $(head -1 "$OUT/$name.$label.$arch.err" | cut -c1-100)"
        fi
    done
}

run_probe /usr/local/cuda-12.9/bin/ptxas cu129 sm_120
run_probe /usr/local/cuda-13.0/bin/ptxas cu130 sm_120
# Anything rejected on plain sm_120: check the 'a' variant (NVFP4 precedent)
run_probe /usr/local/cuda-13.0/bin/ptxas cu130 sm_120a
echo "PROBE_DONE"
