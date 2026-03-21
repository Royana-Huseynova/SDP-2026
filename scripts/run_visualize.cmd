@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..
python -m src.visualize_allclear ^
  --run_dir "results\baseline\uncrtaints\AllClear\test_tx3_s2-s1_100pct_1proi_LOCAL_READY" ^
  --json "external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_LOCAL_READY.json" ^
  --num 3
pause


