#!/bin/bash
# FlashSAC on Isaac-Velocity-*-G1-v0 inside the hssim env (IsaacSim 5.1 / IsaacLab 2.3.0 / torch cu128).
# usage: ./run_g1.sh <exp_name> <num_envs> <num_env_steps> <buffer_max> [extra --overrides ...]
set -e
PY=$HOME/.holosoma_deps/miniconda3/envs/hssim/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$(dirname "$0")"
EXP=$1; NENV=$2; STEPS=$3; BUF=$4; shift 4
ENV_NAME=${ENV_NAME:-Isaac-Velocity-Flat-G1-v0}
$PY train.py --config_name flashSAC_base \
  --overrides group_name=g1 --overrides exp_name=$EXP --overrides seed=0 \
  --overrides env=isaaclab --overrides env.env_name=$ENV_NAME \
  --overrides num_env_steps=$STEPS --overrides num_train_envs=$NENV \
  --overrides num_eval_envs=null --overrides num_record_envs=null \
  --overrides num_eval_episodes=$NENV --overrides num_record_episodes=0 \
  --overrides agent=flashSAC --overrides agent.buffer_max_length=$BUF \
  --overrides agent.buffer_min_length=$((NENV*20)) --overrides agent.sample_batch_size=2048 \
  --overrides agent.use_amp=true --overrides updates_per_interaction_step=2 \
  --overrides agent.asymmetric_observation=false --overrides gamma=0.99 --overrides n_step=3 "$@"
