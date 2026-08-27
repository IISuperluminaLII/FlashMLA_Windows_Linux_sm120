#!/usr/bin/env bash
# ptxas probe: which sm_120 target variant accepts block-scaled NVFP4 MMA (CUDA 13.2 ptxas).
# Control = plain bf16 mma.sync (must pass everywhere; catches PTX syntax false-negatives).
set -u
PTXAS=/usr/local/cuda-13.2/bin/ptxas
WORK=/tmp/bs_probe
mkdir -p "$WORK"

cat > "$WORK/probe_bs.ptx.in" << 'EOF'
.version 9.0
.target TARGET_ARCH
.address_size 64
.visible .entry probe_bs() {
    .reg .b32 ra<4>, rb<2>, sa, sb;
    .reg .f32 d<4>, c<4>;
    mov.b32 ra0, 0; mov.b32 ra1, 0; mov.b32 ra2, 0; mov.b32 ra3, 0;
    mov.b32 rb0, 0; mov.b32 rb1, 0;
    mov.b32 sa, 0; mov.b32 sb, 0;
    mov.f32 c0, 0f00000000; mov.f32 c1, 0f00000000; mov.f32 c2, 0f00000000; mov.f32 c3, 0f00000000;
    mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
        {d0,d1,d2,d3}, {ra0,ra1,ra2,ra3}, {rb0,rb1}, {c0,c1,c2,c3}, sa, {0,0}, sb, {0,0};
    ret;
}
EOF

cat > "$WORK/probe_ctl.ptx.in" << 'EOF'
.version 9.0
.target TARGET_ARCH
.address_size 64
.visible .entry probe_ctl() {
    .reg .b32 ra<4>, rb<2>;
    .reg .f32 d<4>, c<4>;
    mov.b32 ra0, 0; mov.b32 ra1, 0; mov.b32 ra2, 0; mov.b32 ra3, 0;
    mov.b32 rb0, 0; mov.b32 rb1, 0;
    mov.f32 c0, 0f00000000; mov.f32 c1, 0f00000000; mov.f32 c2, 0f00000000; mov.f32 c3, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0,d1,d2,d3}, {ra0,ra1,ra2,ra3}, {rb0,rb1}, {c0,c1,c2,c3};
    ret;
}
EOF

cat > "$WORK/probe_bulkcta.ptx.in" << 'EOF'
.version 9.0
.target TARGET_ARCH
.address_size 64
.visible .entry probe_bulkcta() {
    .reg .b64 rg;
    .shared .align 16 .b8 sbuf[128];
    .shared .align 8 .b8 mbar[8];
    mov.b64 rg, 0;
    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [sbuf], [rg], 64, [mbar];
    ret;
}
EOF

cat > "$WORK/probe_bulkcluster.ptx.in" << 'EOF'
.version 9.0
.target TARGET_ARCH
.address_size 64
.visible .entry probe_bulkcluster() {
    .reg .b64 rg;
    .shared .align 16 .b8 sbuf[128];
    .shared .align 8 .b8 mbar[8];
    mov.b64 rg, 0;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [sbuf], [rg], 64, [mbar];
    ret;
}
EOF

cat > "$WORK/probe_dsm.ptx.in" << 'EOF'
.version 9.0
.target TARGET_ARCH
.address_size 64
.visible .entry probe_dsm() {
    .reg .b32 r1, r2;
    .shared .align 4 .b32 sval;
    mov.b32 r1, sval;
    mapa.shared::cluster.u32 r2, r1, 0;
    ld.shared::cluster.b32 r1, [r2];
    ret;
}
EOF

cat > "$WORK/probe_tcgen05.ptx.in" << 'EOF'
.version 9.0
.target TARGET_ARCH
.address_size 64
.visible .entry probe_tcgen05() {
    .shared .align 4 .b32 taddr;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [taddr], 32;
    ret;
}
EOF

for t in sm_120 sm_120a sm_120f; do
  for p in ctl bs bulkcta bulkcluster dsm tcgen05; do
    sed "s/TARGET_ARCH/$t/" "$WORK/probe_$p.ptx.in" > "$WORK/x.ptx"
    if "$PTXAS" -arch="$t" "$WORK/x.ptx" -o /dev/null 2> "$WORK/err.txt"; then
      echo "[OK]   $t $p"
    else
      echo "[FAIL] $t $p : $(head -c 200 "$WORK/err.txt" | tr '\n' ' ')"
    fi
  done
done
