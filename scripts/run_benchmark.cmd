@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0..\external

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
  --experiment-output-path "%~dp0..\results\baseline\uncrtaints\utae" ^
  --uc-baseline-base-path baselines\UnCRtainTS\model ^
  --uc-weight-folder checkpoints ^
  --uc-exp-name utae ^
  --ctgan-gen-checkpoint "%~dp0..\external\baselines\CTGAN\Pretrain\CTGAN-Sen2_MTC\G_epoch97_PSNR21.259-002.pth" ^
  --utilise-config "%~dp0..\external\baselines\U-TILISE\configs\demo_sen12mscrts.yaml" ^
  --utilise-checkpoint "%~dp0..\external\baselines\U-TILISE\checkpoints\utilise_sen12mscrts_wo_s1.pth"
 

pause