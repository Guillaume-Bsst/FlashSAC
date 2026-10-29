"""Headless scan of the checkpoints of an IsaacLab go2 backflip run (no noise, no randomization, N envs, 1 episode)."""
import os
os.environ["BF_PLAY"] = "1"
import argparse, glob, math, re
import hydra, numpy as np, torch
from omegaconf import OmegaConf
from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env

p = argparse.ArgumentParser(); p.add_argument("--run_dir"); p.add_argument("--num_envs", type=int, default=16); p.add_argument("--only", default=None); p.add_argument("--overrides", action="append", default=[])
a = p.parse_args()
OmegaConf.register_new_resolver("eval", lambda s: eval(s))
hydra.initialize(version_base=None, config_path="./configs")
cfg = hydra.compose(config_name="flashSAC_base", overrides=a.overrides); OmegaConf.resolve(cfg)
env = make_isaaclab_env(env_name=cfg.env.env_name, num_envs=a.num_envs, seed=0, headless=True)
u = env.envs.unwrapped; robot = u.scene["robot"]; sens = u.scene.sensors["contact_forces"]
_, info = env.reset(random_start_init=False)
agent = create_agent(observation_space=env.observation_space, action_space=env.action_space, env_info=info, cfg=cfg.agent)
body_ids = sens.find_bodies(["base", "Head_upper", "Head_lower", ".*_thigh", ".*_calf"])[0]
yax = torch.tensor([0.0, 1.0, 0.0], device=u.device)
from isaaclab.utils.math import quat_apply
head_ids = sens.find_bodies(["Head_upper", "Head_lower"])[0]
for ck in sorted(glob.glob(a.run_dir + "/step*"), key=lambda s: int(re.findall(r"step(\d+)", s)[0])):
    if a.only and os.path.basename(ck) not in a.only.split(","): continue
    agent.load(ck)
    obs, _ = env.reset(random_start_init=False)
    head = torch.zeros(a.num_envs, device=u.device); hpk = torch.zeros(a.num_envs, device=u.device); rot = torch.zeros(a.num_envs, device=u.device); hmax = torch.zeros_like(rot); twist = torch.zeros_like(rot); contact = torch.zeros_like(rot); R = torch.zeros_like(rot)
    for k in range(100):
        act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
        obs, rew, term, trunc, _ = env.step(act)
        d = robot.data
        rot += -d.root_ang_vel_b[:, 1] * u.step_dt; hmax = torch.maximum(hmax, d.root_pos_w[:, 2])
        twist = torch.maximum(twist, (quat_apply(d.root_quat_w, yax.expand(a.num_envs, 3)) - yax).norm(dim=1))
        hf = sens.data.net_forces_w[:, head_ids].norm(dim=-1).max(dim=1).values; head += (hf > 1.0).float(); hpk = torch.maximum(hpk, hf)
        if k > 55: contact += (sens.data.net_forces_w[:, body_ids].norm(dim=-1) > 0.1).any(dim=1).float()
        R += torch.as_tensor(rew, device=u.device)
        if k == 98: fin_z = d.root_pos_w[:, 2].clone(); fin_up = -d.projected_gravity_b[:, 2].clone()
    f = lambda x: f"{float(x.median()):.2f}[{float(x.min()):.2f},{float(x.max()):.2f}]"
    print(f"{os.path.basename(ck)}: ret={f(R)} rot={float(torch.rad2deg(rot).median()):.0f}deg hmax={f(hmax)} final_z={float(fin_z.median()):.2f} up_z={float(fin_up.median()):.2f} contact={float(contact.median()):.0f} twist={f(twist)} HEAD_contact_steps={float(head.median()):.0f} head_peak_force={float(hpk.median()):.0f}N", flush=True)
