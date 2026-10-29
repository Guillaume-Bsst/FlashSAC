#!/bin/bash
# throughput vs num_envs for go2-backflip: each run = 800 interaction steps; log has tqdm timings
cd "$(dirname "$0")"
for cfg in "1024 2" "1024 0" "2048 2" "4096 2" "4096 0"; do
  set -- $cfg; N=$1; U=$2
  ./run_go2_backflip.sh bench_N${N}_U${U} $N $((N*800)) 400000 \
    --overrides updates_per_interaction_step=$U \
    --overrides evaluation_per_interaction_step=null --overrides metrics_per_interaction_step=null \
    --overrides logging_per_interaction_step=100 --overrides save_checkpoint_per_interaction_step=null \
    > logs/bench_N${N}_U${U}.log 2>&1 || echo "FAILED N=$N U=$U"
done
echo BENCH_DONE
