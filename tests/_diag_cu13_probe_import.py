import sys
print("sys.path[0:3]:", sys.path[0:3])
try:
    import flash_mla
    print("IMPORTED:", flash_mla.__file__)
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, e)
