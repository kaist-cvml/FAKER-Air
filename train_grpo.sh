export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 -m aurora.rl.grpo_train \
  --distributed \
  --model aurora \
  --train-start 2016-01-01 --train-end 2020-12-31 \
  --val-start 2023-01-01 --val-end 2023-12-31 \
  --epochs-sft 0 \
  --epochs-grpo 3 \
  --batch 1 \
  --init-ckpt <CHECKPOINTS PATH OF SFT> \
  --ref-ckpt  <CHECKPOINTS PATH OF SFT> \
  --exp-name grpo_rollout_curriculum_cls_reward_KLEveryStep_train2016-2020_lr5e8 \
  --reward cls --reward-temp 0.5 \
  --cls-coarse 0.0 --cls-exact 1.0 --cls-fa-penalty 0.1 \
  --reward-w-pm25 1.0 --reward-w-pm10 0.3 \
  --data-sources obs,cmaq \
  --group-size 4 \
  --rollout-curriculum --rollout-base 3 --rollout-inc 1 --rollout-every 1 \
  --logprob-vars pm2p5,pm10 \
  --logprob-w-pm25 1.0 --logprob-w-pm10 0.3 \
  --kl-w-pm25 1.0 --kl-w-pm10 0.5 \
  --cls-bounds-pm25 0,15,35,75,800 \
  --cls-bounds-pm10 0,30,80,150,1000 \
  --antithetic --common-noise \
  --sigma-pm25 4.0 --sigma-pm10 4.0 \
  --kl-every-step \
  --target-kl 10 --beta-kl 5e-4 --beta-kl-min 1e-6 --beta-kl-max 1e-2 --kp 0.05 --ki 0.01 \
  --adv-ema-coeff 0.1 \
  --lr 1e-7 \
  --sigma-decay 1.0 \
  --amp \
  --hybrid-target \
  --use-masking
  # --kl-every-step
  #   --kl-last-step-only \