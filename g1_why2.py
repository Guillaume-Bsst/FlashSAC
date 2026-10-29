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
sens = u.scene.sensors["contact_forces"]
tm = u.termination_manager
cfg_bc = [c for n, c in zip(tm.active_terms, tm._term_cfgs) if n == "bad_contact"][0]
orig = cfg_bc.func
log = {}
def wrapped(env, **kw):
    r = orig(env, **kw)
    if r.any():
        f = sens.data.net_forces_w.norm(dim=-1)
        for i in torch.nonzero(r)[:, 0].tolist():
            if i in log: continue
            bz = robot.data.body_pos_w[i, :, 2]; bn = robot.body_names
            top = sorted(zip(sens.body_names, f[i].tolist()), key=lambda x: -x[1])[:3]
            log[i] = (int(env.episode_length_buf[i]), [(n, round(v)) for n, v in top], {n: round(float(bz[bn.index(n)]), 2) for n in ("pelvis", "torso_link")}, round(float(bz.min()), 3), round(float(G._torso_up(env)[i]), 2))
    return r
cfg_bc.func = wrapped
for k in range(100):
    act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
    obs, rew, term, trunc, _ = env.step(act)
for i, v in sorted(log.items())[:6]:
    print("TERM env", i, "ep_step", v[0], "top_forces", v[1], "heights", v[2], "min_body_z", v[3], "up", v[4])
print("TERM n_envs_terminated", len(log))
