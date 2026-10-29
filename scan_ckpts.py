"""Headless scan: 1 deterministic episode per checkpoint of a Genesis go2-backflip run."""
import argparse, glob, math, os, re
import hydra, numpy as np, torch
from omegaconf import OmegaConf
import play_genesis as pg
from flash_rl.agents import create_agent
from flash_rl.envs.genesis import make_genesis_env
from flash_rl.envs.genesis_envs import go2_backflip

p = argparse.ArgumentParser(); p.add_argument("--run_dir"); p.add_argument("--overrides", action="append", default=[])
a = p.parse_args()
OmegaConf.register_new_resolver("eval", lambda s: eval(s))
hydra.initialize(version_base=None, config_path="./configs")
cfg = hydra.compose(config_name="flashSAC_base", overrides=a.overrides); OmegaConf.resolve(cfg)
pg.SHOW_VIEWER = False
go2_backflip.get_env = pg._get_env_with_viewer
env = make_genesis_env(env_name=cfg.env.env_name, num_envs=1, rescale_action=cfg.env.rescale_action, eval_mode=True)
_, info = env.reset()
agent = create_agent(observation_space=env.observation_space, action_space=env.action_space, env_info=info, cfg=cfg.agent)
base = env.base_env; dt = float(base.dt)
for ck in sorted(glob.glob(a.run_dir + "/step*"), key=lambda s: int(re.findall(r"step(\d+)", s)[0])):
    agent.load(ck); obs, _ = env.reset()
    twist = 0.0; rot = hmax = 0.0; contact = 0; done = False; n = 0; R = 0.0
    while not done:
        act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
        obs, rew, term, trunc, _ = env.step(act); n += 1; R += float(np.asarray(rew).reshape(-1)[0])
        yw = go2_backflip.gs_transform_by_quat(torch.tensor([[0.0, 1.0, 0.0]], device=base.device), base.base_quat[:1]); twist = max(twist, float((yw - torch.tensor([[0.0, 1.0, 0.0]], device=base.device)).norm()))
        rot += float(base.base_ang_vel[0, 1]) * dt; hmax = max(hmax, float(base.base_pos[0, 2]))
        if n > 55: contact += int((base.link_contact_forces[0, base.penalized_contact_link_indices, :].norm(dim=-1) > 0.1).any())
        done = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
    print(f"{os.path.basename(ck)}: return={R:.1f} rot={math.degrees(rot):.0f} hmax={hmax:.2f} final_h={float(base.base_pos[0,2]):.2f} up_z={-float(base.projected_gravity[0,2]):.2f} contact={contact} twist_max={twist:.2f}", flush=True)
