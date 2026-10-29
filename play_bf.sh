#!/bin/bash
cd ~/FlashSAC
for s in 6000 12000; do
  echo "=== checkpoint step$s ==="
  DISPLAY=:1 ~/.flashsac_env/bin/python play_genesis.py --checkpoint_path models/go2bf/go2bf_A_4096_u2/go2-backflip/seed0-1021-173640/step$s --num_episodes 6 --slowmo 3 \
    --overrides env=genesis --overrides env.env_name=go2-backflip --overrides agent=flashSAC \
    --overrides agent.asymmetric_observation=true --overrides agent.buffer_max_length=1 --overrides agent.buffer_min_length=1 \
    --overrides agent.buffer_device_type=cuda --overrides gamma=0.95 --overrides n_step=1 2>&1 | grep -v -i warn
done
echo PLAY_DONE
