import json
import os
from pathlib import Path

SRC = Path(r"external\metadata\datasets\test_on_dataset_root_EXISTING.json")
DST = Path(r"external\metadata\datasets\test_on_dataset_root_EXISTING_DW.json")

def s2_to_dw_path(s2_path: str) -> str:
    # Works because your filenames match: roiXXXX_s2_toa_YYYY_M_DD_median.tif -> roiXXXX_dw_YYYY_M_DD_median.tif
    return s2_path.replace("\\s2_toa\\", "\\dw\\").replace("_s2_toa_", "_dw_")

def main():
    data = json.load(open(SRC, "r", encoding="utf-8"))

    kept, dropped = 0, 0
    out = {}

    for k, v in data.items():
        # Use TARGET S2 path to derive expected DW path
        target_s2 = v["target"][0][1]
        dw_path = s2_to_dw_path(target_s2)

        if os.path.exists(dw_path):
            out[k] = v
            kept += 1
        else:
            dropped += 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DST, "w", encoding="utf-8"), indent=2)

    print("Wrote:", DST)
    print("Kept:", kept, "Dropped:", dropped)

if __name__ == "__main__":
    main()