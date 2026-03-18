import json
from pathlib import Path
import os

def fix_path(p: str) -> str:
    pn = str(p).replace("\\", "/")
    pn = pn.replace("/external/allclear/allclear_dataset/", "/external/dataset/")
    pn = pn.replace("/external/allclear_dataset/", "/external/dataset/")
    return pn.replace("/", "\\")

def main():
    src = Path("external/metadata/datasets/test_tx3_s2-s1_100pct_1proi_local.json")
    dst = Path("external/metadata/datasets/test_on_dataset_root_EXISTING.json")

    data = json.loads(src.read_text(encoding="utf-8"))
    out = {}
    kept = 0
    dropped = 0

    for k, v in data.items():
        v2 = dict(v)

        # rewrite
        v2["s2_toa"] = [[ts, fix_path(p)] for ts, p in v2["s2_toa"]]
        if "s1" in v2:
            v2["s1"] = [[ts, fix_path(p)] for ts, p in v2["s1"]]
        if "target" in v2:
            v2["target"] = [[ts, fix_path(p)] for ts, p in v2["target"]]

        # keep only if the target file exists
        target_path = v2["target"][0][1]
        if os.path.exists(target_path):
            out[k] = v2
            kept += 1
        else:
            dropped += 1

    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote:", dst.resolve())
    print("Kept:", kept, "Dropped:", dropped)

if __name__ == "__main__":
    main()