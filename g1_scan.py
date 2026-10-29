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
for ck in sorted(glob.glob(a.run_dir + "/step*"), key=lambda s: int(re.findall(r"step(\d+)", s)[0])):
    if a.only and os.path.basename(ck) not in a.only.split(","): continue
    agent.load(ck)
    obs, _ = env.reset(random_start_init=False)
    n = a.num_envs; alive = torch.ones(n, dtype=torch.bool, device=u.device)
    held = torch.zeros(n, device=u.device); upmin = torch.ones(n, device=u.device); zmax = torch.zeros(n, device=u.device); steps = torch.zeros(n, device=u.device)
    ret = torch.zeros(n, device=u.device); upfin = torch.zeros(n, device=u.device); zfin = torch.zeros(n, device=u.device)
    for k in range(T):
        act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
        obs, rew, term, trunc, _ = env.step(act)
        up = G._torso_up(u); z = robot.data.root_pos_w[:, 2]
        held += (alive & (G.hold_bonus(u) > 0)).float() * u.step_dt
        upmin = torch.where(alive, torch.minimum(upmin, up), upmin); zmax = torch.where(alive, torch.maximum(zmax, z), zmax)
        steps += alive.float(); ret += alive.float() * torch.as_tensor(rew, device=u.device)
        upfin = torch.where(alive, up, upfin); zfin = torch.where(alive, z, zfin)
        alive &= ~torch.as_tensor(np.logical_or(term, trunc), device=u.device)
    f = lambda x: f"{float(x.median()):.2f}[{float(x.min()):.2f},{float(x.max()):.2f}]"
    print(f"{os.path.basename(ck)}: ret={f(ret)} held_s={f(held)} best_held={float(held.max()):.2f} up_min={f(upmin)} pelvis_zmax={f(zmax)} survive_s={float(steps.median()*u.step_dt):.2f} frac_held>1s={float((held>1).float().mean()):.2f}", flush=True)
