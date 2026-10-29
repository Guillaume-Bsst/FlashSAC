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
names = sens.body_names
tm = u.termination_manager
print("TERM terms", tm.active_terms)
first = {}
for k in range(T):
    act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
    f_before = None
    obs, rew, term, trunc, _ = env.step(act)
    done = torch.as_tensor(np.logical_or(term, trunc), device=u.device)
    if done.any() and k < 400:
        for i in torch.nonzero(done)[:3, 0].tolist():
            if i in first: continue
            first[i] = k
            reasons = {t: bool(tm.get_term(t)[i]) for t in tm.active_terms}
            f = sens.data.net_forces_w[i].norm(dim=-1)
            top = sorted(zip(names, f.tolist()), key=lambda x: -x[1])[:4]
            bz = robot.data.body_pos_w[i, :, 2]; bn = robot.body_names
            tz = {n: round(float(bz[bn.index(n)]), 2) for n in ("pelvis", "torso_link", "right_shoulder_yaw_link", "left_ankle_roll_link")}
            print(f"TERM env{i} step{k} t={k*u.step_dt:.2f}s reasons={reasons} top_contacts={[(n, round(v,1)) for n, v in top]} heights={tz} min_body_z={float(bz.min()):.3f} up={float(G._torso_up(u)[i]):+.2f} pz={float(robot.data.root_pos_w[i,2]):.2f}")
