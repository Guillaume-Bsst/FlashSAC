#!/bin/bash
# E0 : rejeu en boucle ouverte pour chaque référence (σ = 0, σ plancher) et chaque k, l'un après l'autre.
set -e
cd "$(dirname "$0")/.."
rm -f results/ant_lab/e0/openloop.jsonl
for ref in ref_s0 ref_s1; do
  for k in 1.0 1.25 1.5; do
    echo "E0 : $ref k=$k"
    ANT_LAB_REF=results/ant_lab/refs/$ref.npz ANT_LAB_K=$k ant_lab/run.sh -m ant_lab.e0_openloop \
      > results/ant_lab/logs/e0_${ref}_k$k.log 2>&1
    grep "e0_openloop :" results/ant_lab/logs/e0_${ref}_k$k.log
  done
done
