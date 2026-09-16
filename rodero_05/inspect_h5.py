import sys, json, glob
import h5py
import numpy as np

fname = sys.argv[1]
with h5py.File(fname, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:50s} shape={obj.shape} dtype={obj.dtype}")
        else:
            attrs = dict(obj.attrs)
            print(f"{name}/  {attrs if attrs else ''}")
    f.visititems(show)

    for key in ("mesh/coordinates", "geometry/mesh/coordinates"):
        if key in f:
            x = f[key][:]
            print(f"\n{key} bounding box:")
            print("  min", x.min(axis=0))
            print("  max", x.max(axis=0))
            print("  extent", x.max(axis=0) - x.min(axis=0))

for j in glob.glob("*.json"):
    print(f"\n--- {j} ---")
    print(json.dumps(json.load(open(j)), indent=2))
