CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_addr="127.0.0.1" \
  --master_port=29507 \
  train.py --batch 2 \
  --model aurora \
  --data-sources obs,cmaq \
  --cmaq-root ./data/cmaq_only_npy \
  --obs-root ./data/obs_npz_27km \
  --epochs 30 \
  --exp-name obs27+cmaq_with_aurora25 \
  --train-start 2016-01-01 --train-end 2022-12-31 \
  --val-start 2023-01-01 --val-end 2023-03-31 \
  --rollout-steps 1 \
  --accum-steps 1 \
  --w-cmaq 0.5 \
 --use_cmaq_pm_only \
 --use_hybrid_target \
 --use-masking \
 --w-pm25-good 0.2 \
 --w-pm25-moderate 0.2 \
 --w-pm25-bad 1.0 \
 --w-pm25-very-bad 0.5 \
 --use_cutmix
#  --use_hybrid_target \
#  --use_cutmix \
#  --use-masking
#  --pattern_aware_loss