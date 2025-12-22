CUDA_VISIBLE_DEVICES=0 \
torchrun \
  --nproc_per_node=1 \
  --master_addr="127.0.0.1" \
  --master_port=29502 \
  test.py --batch 1 \
  --model aurora \
  --test-start-date 2023-01-01 \
  --test-end-date 2023-12-31 \
  --data-sources obs,cmaq \
  --checkpoint-path checkpoints_grpo/Train:2016-01-01-2022-12-31_Test:2023-01-01-2023-03-31/obs27+cmaq_with_aurora25_accstep1_use_cmaq_pm_only_cls_weight_grpo_rollout4_init_2021-2023_class_reward_klEveryStep_hybrid/best_policy.pth \
  --npz-path ./data/23-air-pollution/obs_npz_27km \
  --cmaq-root ./data/cmaq_only_npy \
  --mode rollout \
  --rollout-hours 120 \
  --use_cmaq_pm_only
#   --use_cutmix
# --use_hybrid_target
#  --use_cutmix
# --use-wind-prompt