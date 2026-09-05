#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
FLASH_MLA_SM120_BWD_MMA_CFG=2 "$PY" tests/_probe_bwd_cfg6_dq.py /tmp/_dq_cfg2 2>&1 | grep -E 'CFG=|Error|error'
FLASH_MLA_SM120_BWD_MMA_CFG=6 "$PY" tests/_probe_bwd_cfg6_dq.py /tmp/_dq_cfg6 2>&1 | grep -E 'CFG=|Error|error'
"$PY" - <<'PY'
import torch
for d in (128, 192):
    a = torch.load(f"/tmp/_dq_cfg2_{d}.pt"); b = torch.load(f"/tmp/_dq_cfg6_{d}.pt")
    print(f"{d}: corr(flat)={torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0,1]:.4f}")
    # is b a row-permutation / col-permutation of a? check per-row and per-col energy
    ra = a.pow(2).sum(dim=(1,2)); rb = b.pow(2).sum(dim=(1,2))
    print(f"   row energy match (first 8 rows) a={ra[:8].numpy().round(2)} b={rb[:8].numpy().round(2)}")
    # find, for the first few rows of b, the best matching row of a
    for r in range(4):
        sims = torch.nn.functional.cosine_similarity(b[r].flatten().unsqueeze(0), a.flatten(1), dim=1)
        j = int(sims.argmax()); print(f"   b row {r} best matches a row {j} (cos {sims[j]:.3f})")
PY
echo "DQ_PROBE_DONE"
