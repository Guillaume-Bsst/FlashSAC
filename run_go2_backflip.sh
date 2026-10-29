#!/bin/bash
# FlashSAC on Genesis go2-backflip (isolated env ~/.flashsac_env: torch 2.9.1+cu128, genesis-world 0.3.1).
# usage: ./run_go2_backflip.sh <exp_name> <num_envs> <num_env_steps> <buffer_max> [extra --overrides ...]
set -e
PY=$HOME/.flashsac_env/bin/python
cd "$(dirname "$0")"
EXP=$1; NENV=$2; STEPS=$3; BUF=$4; shift 4
$PY train.py --config_name flashSAC_base \
  --overrides group_name=go2bf --overrides exp_name=$EXP --overrides seed=0 \
  --overrides env=genesis --overrides env.env_name=go2-backflip \
  --overrides num_env_steps=$STEPS --overrides num_train_envs=$NENV \
  --overrides num_eval_envs=null --overrides num_record_envs=null \
  --overrides num_eval_episodes=$NENV --overrides num_record_episodes=0 \
  --overrides agent=flashSAC --overrides agent.buffer_max_length=$BUF \
  --overrides agent.buffer_min_length=$((NENV*20)) --overrides agent.buffer_device_type=cuda \
  --overrides agent.sample_batch_size=2048 --overrides agent.use_amp=true \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.asymmetric_observation=true --overrides gamma=0.95 --overrides n_step=1 "$@"
