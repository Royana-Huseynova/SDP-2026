@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..
python -m src.visualize_allclear ^
  --run_dir "results\baseline\uncrtaints\AllClear\test_tx3_s2-s1_100pct_1proi_local" ^
  --json "external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_local.json" ^
  --num 100
pause


