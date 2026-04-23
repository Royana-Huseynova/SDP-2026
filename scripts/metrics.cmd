@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..
python src\metrics.py ^
  --run_dir "C:\Users\User\Desktop\SDP\allclear_test_proi1_v1\SDP-2026-1\results\baseline\uncrtaints\utae\AllClear\test_tx3_s2-s1_100pct_1proi_local" ^
  --model uncrtaints ^
  --json "external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_local.json"

pause