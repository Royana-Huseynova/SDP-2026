import json
import os
from pathlib import Path

# =========================
# CONFIG
# =========================
SRC_JSON = Path(r"C:\Users\User\Desktop\SDP\allclear_test_proi1_v1\SDP-2026-1\external\metadata\datasets\test_tx3_s2-s1_100pct_1proi.json")
DST_JSON = Path(r"C:\Users\User\Desktop\SDP\allclear_test_proi1_v1\SDP-2026-1\external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_LOCAL_READY.json")

# Folder that contains roi folders
LOCAL_DATA_ROOT = Path(r"C:\Users\User\Desktop\SDP\allclear_test_proi1_v1\SDP-2026-1\external\dataset")

# True = drop entries if any referenced file is missing
FILTER_MISSING_FILES = True

# True = also require DW file for the target S2 image
FILTER_MISSING_DW = True

# True = print reasons when dropping entries
VERBOSE = True
# =========================


def extract_relative_from_old_path(old_path: str) -> Path:
    """
    Convert a source path like:
    /share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/roi30630/2022_8/s2_toa/file.tif

    into:
    roi30630/2022_8/s2_toa/file.tif
    """
    p = str(old_path).replace("\\", "/")

    marker = "/dataset_30k_v4/"
    if marker in p:
        rel = p.split(marker, 1)[1]
        return Path(*rel.split("/"))

    parts = p.split("/")
    roi_idx = None
    for i, part in enumerate(parts):
        if part.startswith("roi"):
            roi_idx = i
            break

    if roi_idx is None:
        raise ValueError(f"Could not find ROI in path: {old_path}")

    rel = parts[roi_idx:]
    return Path(*rel)


def convert_old_path_to_local(old_path: str) -> str:
    rel = extract_relative_from_old_path(old_path)
    return str(LOCAL_DATA_ROOT / rel)


def fix_sequence(seq):
    """
    Input format is expected like:
    [
        ["2022-03-18", "/share/.../file.tif"],
        ...
    ]
    """
    fixed = []
    for item in seq:
        if not isinstance(item, list) or len(item) != 2:
            fixed.append(item)
            continue

        ts, old_path = item
        new_path = convert_old_path_to_local(old_path)
        fixed.append([ts, new_path])

    return fixed


def s2_to_dw_path(s2_path: str) -> str:
    return s2_path.replace(f"{os.sep}s2_toa{os.sep}", f"{os.sep}dw{os.sep}").replace("_s2_toa_", "_dw_")


def all_files_exist(sample: dict):
    missing = []

    for key in ("s2_toa", "s1", "target"):
        if key not in sample:
            continue

        for item in sample[key]:
            if isinstance(item, list) and len(item) == 2:
                _, fpath = item
                if not os.path.exists(fpath):
                    missing.append(fpath)

    return len(missing) == 0, missing


def dw_exists_for_target(sample: dict):
    if "target" not in sample or not sample["target"]:
        return False, None

    target_s2 = sample["target"][0][1]
    dw_path = s2_to_dw_path(target_s2)

    return os.path.exists(dw_path), dw_path


def main():
    if not SRC_JSON.exists():
        raise FileNotFoundError(f"SRC_JSON not found: {SRC_JSON}")

    if not LOCAL_DATA_ROOT.exists():
        raise FileNotFoundError(f"LOCAL_DATA_ROOT not found: {LOCAL_DATA_ROOT}")

    with open(SRC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    kept = 0
    dropped_missing = 0
    dropped_dw = 0
    dropped_other = 0

    for sample_id, sample in data.items():
        try:
            fixed = dict(sample)

            if "s2_toa" in fixed:
                fixed["s2_toa"] = fix_sequence(fixed["s2_toa"])

            if "s1" in fixed:
                fixed["s1"] = fix_sequence(fixed["s1"])

            if "target" in fixed:
                fixed["target"] = fix_sequence(fixed["target"])

            if FILTER_MISSING_FILES:
                exists_ok, missing_files = all_files_exist(fixed)
                if not exists_ok:
                    dropped_missing += 1
                    if VERBOSE:
                        print(f"[DROP missing files] {sample_id}")
                        for mf in missing_files[:10]:
                            print("   ", mf)
                        if len(missing_files) > 10:
                            print(f"    ... and {len(missing_files) - 10} more")
                    continue

            if FILTER_MISSING_DW:
                dw_ok, dw_path = dw_exists_for_target(fixed)
                if not dw_ok:
                    dropped_dw += 1
                    if VERBOSE:
                        print(f"[DROP missing DW] {sample_id}")
                        print("   ", dw_path)
                    continue

            out[sample_id] = fixed
            kept += 1

        except Exception as e:
            dropped_other += 1
            if VERBOSE:
                print(f"[DROP error] {sample_id}: {e}")

    DST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DST_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== DONE ===")
    print("Source JSON :", SRC_JSON)
    print("Output JSON :", DST_JSON)
    print("Total input :", len(data))
    print("Kept        :", kept)
    print("Dropped missing files :", dropped_missing)
    print("Dropped missing DW    :", dropped_dw)
    print("Dropped other errors  :", dropped_other)


if __name__ == "__main__":
    main()