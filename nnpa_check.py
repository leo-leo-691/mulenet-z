"""Evidence that Snap ML's zdnn_tensors path really goes through IBM's zDNN library on the
on-chip AI accelerator: is libzdnn loaded into the process, and does it report NNPA?"""
import ctypes
import numpy as np
from snapml import BoostingMachineClassifier

rng = np.random.default_rng(0)
X = rng.random((2000, 10), dtype=np.float32)
y = (X[:, 0] > 0.5).astype(np.float32)
bm = BoostingMachineClassifier(num_round=20, max_depth=4, random_state=42)
bm.fit(X, y)
bm.optimize_trees(tree_format="zdnn_tensors")
bm.predict_proba(X[:100])

libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "zdnn" in l.lower() and "/" in l})
print("zDNN libraries loaded in this process:", libs or "none")
for path in libs:
    try:
        lib = ctypes.CDLL(path)
        lib.zdnn_is_nnpa_installed.restype = ctypes.c_bool
        print(f"{path}: zdnn_is_nnpa_installed() = {lib.zdnn_is_nnpa_installed()}")
        if hasattr(lib, "zdnn_get_library_version_str"):
            lib.zdnn_get_library_version_str.restype = ctypes.c_char_p
            print("zDNN version:", lib.zdnn_get_library_version_str().decode())
    except Exception as e:
        print(f"{path}: could not query ({type(e).__name__}: {e})")
