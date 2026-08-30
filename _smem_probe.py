import torch, ctypes
torch.cuda.init()
p = torch.cuda.get_device_properties(0)
print("device:", p.name)
for a in dir(p):
    if "shared" in a.lower() or "smem" in a.lower():
        print(" ", a, "=", getattr(p, a))
# direct CUDA runtime query
cudart = ctypes.CDLL("cudart64_12.dll")
val = ctypes.c_int(0)
# cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 ; cudaDevAttrMaxSharedMemoryPerBlock = 8
for name, attr in [("MaxSharedMemPerBlock", 8), ("MaxSharedMemPerBlockOptin", 97), ("MaxSharedMemPerMultiprocessor", 81)]:
    r = cudart.cudaDeviceGetAttribute(ctypes.byref(val), attr, 0)
    print(f"  {name}: {val.value} bytes ({val.value/1024:.0f} KB)  [rc={r}]")
