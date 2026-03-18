@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0
python visualize_allclear.py ^
  --run_dir "results\baseline\uncrtaints\AllClear\test_on_dataset_root_EXISTING_DW" ^
  --json "external\metadata\datasets\test_on_dataset_root_EXISTING_DW.json" ^
  --num 54
pause


