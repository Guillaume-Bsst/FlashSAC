# G1 forearm stand (inverted balance) — design

## Goal
Unitree G1 (29 DoF) starts standing, gets onto its forearms and holds a stable inverted balance
(legs vertical, pelvis above the forearm support), similar to MOYO yoga poses (Pincha Mayurasana).
Trained with FlashSAC in IsaacLab, then sim2sim in MuJoCo with the GR00T deploy MJCF. Sim only, no real robot.

## Decisions (user)
- No reference motion: rewards + hand-written target pose (no MOYO data on this machine).
- Robot: G1 29 DoF, real torque limits (legs 88/139 Nm, ankles/waist 50, arms 25, wrist pitch/yaw 5).
- The inversion is dynamic: legs push (kick-up), arms mostly support and catch.

## Phases (each trained from its own reset distribution)
1. Kick-up: start in "dolphin" pose (forearms on ground, hips high, knees bent). Legs extend, pelvis goes over the forearms.
2. Catch/balance: absorb the momentum near vertical, hold the pose without falling over the other side.
3. Approach: standing -> lean forward -> forearms on the ground (dolphin). Chained last.

Curriculum: train 1+2 first from the dolphin pose; then mix reset states (dolphin / near-vertical with angular velocity /
standing) and add phase 3. Reset states may later be harvested from successful rollouts.

## Env (IsaacLab, FlashSAC, `flash_rl/envs/g1_forearm_stand_isaaclab_env.py`)
- Observations: base ang vel, projected gravity, joint pos/vel, last action, time phase. No base linear velocity.
- Rewards: pelvis height above support, trunk axis vertical (inverted), legs vertical, survival while inverted,
  pelvis velocity penalty, head/hand contact penalty, torque + action-rate penalties from the start.
- Contacts allowed: forearms (elbow_link, wrist_roll_link); penalized: head/torso/hands/feet-on-ground in the final pose.
- Termination: head/torso touches ground, or tilt beyond threshold.
- Actions: joint targets around default, PD with real gains/limits.

## Evaluation
IsaacLab scan (time inverted, support drift, peak contact forces), then MuJoCo sim2sim with the GR00T MJCF.
Success: stays inverted >= 5 s without head/torso contact.

## Risks
- Arm torque (25 Nm) may not suffice for the push/catch; static hold should be feasible with vertical upper arms.
- Forearm collision geometry may differ between IsaacLab (USD) and MuJoCo (MJCF).
- The kick-up may overshoot; the catch phase may stay unlearned.
- Backflip lesson: wide randomization or heavy penalties early can prevent the skill from being learned; add them progressively.

## Pre-training checks
Joint names/limits in the USD, forearm collision shapes, existence of a stable dolphin pose within limits.
