@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0external

python -m allclear.benchmark ^
  --dataset-fpath "%~dp0..\external\metadata\datasets\test_tx3_s2-s1_100pct_1proi_local.json" ^
  --model-name uncrtaints ^
  --device cpu ^
  --main-sensor s2_toa ^
  --aux-sensors s1 ^
  --aux-data cld_shdw dw ^
  --target-mode s2p ^
  --tx 3 ^
  --batch-size 1 ^
  --num-workers 0 ^
  --draw-vis 0 ^
  --experiment-output-path "%~dp0..\results\baseline\uncrtaints" ^
  --uc-baseline-base-path baselines\UnCRtainTS\model ^
  --uc-weight-folder checkpoints ^
  --uc-exp-name noSAR_1 

pause