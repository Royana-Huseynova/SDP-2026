@echo off
REM ===================================================================
REM SDP-2026 unified CLI launcher (Windows)
REM
REM Usage:
REM   scripts\sdp.cmd describe
REM   scripts\sdp.cmd benchmark --dataset allclear --variant uncrtaints ^
REM         --device cpu --dataset-fpath external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_local.json
REM ===================================================================
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..
python -m sdp %*
endlocal
