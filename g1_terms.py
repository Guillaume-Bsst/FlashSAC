"""Headless scan of a G1 forearm-stand run: inverted score, time held, pelvis height, contacts (no noise, N envs)."""
import os
os.environ["BF_PLAY"] = "1"
import argparse, glob, re
import hydra, numpy as np, torch
from omegaconf import OmegaConf
from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env

p = argparse.ArgumentParser(); p.add_argument("--run_dir"); p.add_argument("--num_envs", type=int, default=32); p.add_argument("--only", default=None); p.add_argument("--overrides", action="append", default=[])
a = p.parse_args()
OmegaConf.register_new_resolver("eval", lambda s: eval(s))
hydra.initialize(version_base=None, config_path="./configs")
cfg = hydra.compose(config_name="flashSAC_base", overrides=a.overrides); OmegaConf.resolve(cfg)
env = make_isaaclab_env(env_name=cfg.env.env_name, num_envs=a.num_envs, seed=0, headless=True)
import flash_rl.envs.g1_forearm_stand_isaaclab_env as G
u = env.envs.unwrapped; robot = u.scene["robot"]
_, info = env.reset(random_start_init=False)
agent = create_agent(observation_space=env.observation_space, action_space=env.action_space, env_info=info, cfg=cfg.agent)
T = int(u.max_episode_length)
ck = glob.glob(a.run_dir + "/" + a.only)[0]; agent.load(ck)
obs, _ = env.reset(random_start_init=False)
acc = {}; n_acc = 0
for k in range(T):
    act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
    obs, rew, term, trunc, _ = env.step(act)
    if k >= T - 150:
        for name, cfg in zip(u.reward_manager.active_terms, u.reward_manager._term_cfgs):
            v = (cfg.func(u, **cfg.params) * cfg.weight).mean().item()
            acc[name] = acc.get(name, 0) + v
        for nm, v in (("up", G._torso_up(u)), ("legs_dir", G._legs_dir_z(u)), ("pelvis_z", robot.data.root_pos_w[:, 2])):
            acc[nm] = acc.get(nm, 0) + v.mean().item()
        for i, nm in enumerate(("L_elbow", "L_wrist", "R_elbow", "R_wrist")):
            acc[nm] = acc.get(nm, 0) + G._fore4(u)[:, i].mean().item()
        n_acc += 1
tot = 0
for k, v in acc.items():
    print(f"TERM {k:14s} {v / n_acc:8.3f}")
print("TERM total_reward", sum(v / n_acc for k, v in acc.items() if k in u.reward_manager.active_terms))
