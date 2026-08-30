#!/usr/bin/env bash
# Probe Blackwell 256-bit (32-byte) vector load/store paths on plain sm_120:
# CUDA 13's 32B-aligned vector types map to ld/st.global.v8.f32 (and .v4.f64).
# Also confirm cp.async stays capped at 16B (32B expected REJECT). Compile-only.
set -u
OUT=/tmp/_bw_vec_probes
mkdir -p "$OUT"

emit() {  # emit <name> <ver> <body>
    local name="$1" ver="$2"; shift 2
    cat > "$OUT/$name.ptx" <<EOF
.version $ver
.target sm_120
.address_size 64
.visible .global .align 32 .f32 gbuf[16];
.visible .entry probe_$name()
{
    .reg .f32 %f<10>;
    .reg .f64 %d<6>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .shared .align 32 .b8 sbuf[256];
    mov.u64 %rd1, gbuf;
    mov.f32 %f1, 0f3F800000;
    $*
    ret;
}
EOF
}

BODY_LDG_V8='ld.global.nc.v8.f32 {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8}, [%rd1];'
BODY_STG_V8='st.global.v8.f32 [%rd1], {%f1,%f1,%f1,%f1,%f1,%f1,%f1,%f1};'
BODY_LDG_V4D='ld.global.v4.f64 {%d1,%d2,%d3,%d4}, [%rd1];'
BODY_LDS_V8='mov.u32 %r1, sbuf; ld.shared.v8.f32 {%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8}, [%r1];'
BODY_STS_V8='mov.u32 %r1, sbuf; st.shared.v8.f32 [%r1], {%f1,%f1,%f1,%f1,%f1,%f1,%f1,%f1};'
BODY_CPA_32='mov.u32 %r1, sbuf; cp.async.cg.shared.global [%r1], [%rd1], 32;'

for ver in 8.7 9.0; do
    tag="${ver/./}"
    emit "ldg_v8_f32_$tag"  "$ver" "$BODY_LDG_V8"
    emit "stg_v8_f32_$tag"  "$ver" "$BODY_STG_V8"
    emit "ldg_v4_f64_$tag"  "$ver" "$BODY_LDG_V4D"
    emit "lds_v8_f32_$tag"  "$ver" "$BODY_LDS_V8"
    emit "sts_v8_f32_$tag"  "$ver" "$BODY_STS_V8"
    emit "cpasync_32B_$tag" "$ver" "$BODY_CPA_32"
done

run_probe() {  # <ptxas> <label> <verfilter>
    local PTXAS="$1" label="$2" vf="$3"
    echo "=== $label (PTX $vf) arch=sm_120 ==="
    for f in "$OUT"/*_"${vf/./}".ptx; do
        local name; name="$(basename "$f" .ptx)"
        if "$PTXAS" -arch=sm_120 -o /dev/null "$f" 2>"$OUT/$name.$label.err"; then
            echo "  [ACCEPT] $name"
        else
            echo "  [REJECT] $name :: $(head -1 "$OUT/$name.$label.err" | cut -c1-90)"
        fi
    done
}

run_probe /usr/local/cuda-12.9/bin/ptxas cu129 8.7
run_probe /usr/local/cuda-13.0/bin/ptxas cu130 9.0
echo "PROBE_DONE"
