@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0external

python -m allclear.benchmark ^
  --dataset-fpath "%~dp0external\metadata\datasets\test_on_dataset_root_EXISTING_DW.json" ^
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
  --experiment-output-path "%~dp0results\baseline\uncrtaints" ^
  --uc-baseline-base-path baselines\UnCRtainTS\model ^
  --uc-weight-folder checkpoints ^
  --uc-exp-name noSAR_1

pause