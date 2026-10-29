import os, sys
os.environ["BF_PLAY"] = "1"; sys.path.insert(0, "/home/ecarn/FlashSAC")
import glob, math, hydra, numpy as np, torch
from omegaconf import OmegaConf
from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env
OmegaConf.register_new_resolver("eval", lambda s: eval(s))
hydra.initialize(version_base=None, config_path="./configs")
cfg = hydra.compose(config_name="flashSAC_base", overrides=["env=isaaclab","env.env_name=FlashSAC-Backflip-Go2-v0","agent=flashSAC","agent.asymmetric_observation=false","agent.buffer_max_length=1","agent.buffer_min_length=1","agent.buffer_device_type=cuda","gamma=0.95","n_step=1"]); OmegaConf.resolve(cfg)
env = make_isaaclab_env(env_name=cfg.env.env_name, num_envs=1, seed=0, headless=True)
u = env.envs.unwrapped; r = u.scene["robot"]; d = r.data
_, info = env.reset(random_start_init=False)
agent = create_agent(observation_space=env.observation_space, action_space=env.action_space, env_info=info, cfg=cfg.agent)
agent.load(sys.argv[1]); obs, _ = env.reset(random_start_init=False)
rot = 0.0; vmax = torch.zeros(12, device=u.device); tqmax = torch.zeros(12, device=u.device)
rows = []
sens = u.scene.sensors['contact_forces']; hid = sens.find_bodies(['Head_upper','Head_lower'])[0]; bid = sens.find_bodies(['base'])[0]; fid = sens.find_bodies(['.*_foot'])[0]; tid = sens.find_bodies(['.*_thigh','.*_calf'])[0]
cont = []
slip_len = 0.0
rec = {k: [] for k in ('act','qpos','qvel','z','x','rot','tq')}
for k in range(100):
    act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
    obs, rew, term, trunc, _ = env.step(act)
    rot += -float(d.root_ang_vel_b[0, 1]) * u.step_dt
    vmax = torch.maximum(vmax, d.joint_vel[0].abs()); tqmax = torch.maximum(tqmax, d.applied_torque[0].abs())
    rec['act'].append(np.clip(act.reshape(-1),-1,1)); rec['qpos'].append(d.joint_pos[0].cpu().numpy()); rec['qvel'].append(d.joint_vel[0].cpu().numpy()); rec['z'].append(float(d.root_pos_w[0,2])); rec['x'].append(float(d.root_pos_w[0,0])); rec['rot'].append(math.degrees(rot)); rec['tq'].append(d.applied_torque[0].cpu().numpy())
    nf = sens.data.net_forces_w[0]
    if k < 25:
        inc = nf[fid].norm(dim=-1) > 1.0
        vf = d.body_lin_vel_w[0][r.find_bodies('.*_foot')[0]][:, :2].norm(dim=-1)
        slip_len += float((vf * inc).sum()) * u.step_dt
    cont.append((k, float(nf[hid].norm(dim=-1).max()), float(nf[bid].norm(dim=-1).max()), int((nf[fid].norm(dim=-1) > 1).sum()), float(nf[tid].norm(dim=-1).max()), math.degrees(rot), float(d.root_pos_w[0,2])))
    rows.append([k, float(d.root_pos_w[0,0]), float(d.root_pos_w[0,2]), math.degrees(rot), float(d.root_lin_vel_w[0,2]), float(d.joint_vel[0, 8:].abs().max())])
print("ISO joint order", r.joint_names)
print("ISO max |joint_vel| per joint", [round(float(x),1) for x in vmax])
print("ISO max |applied torque| per joint", [round(float(x),1) for x in tqmax])
for row in rows[::6]: print("ISO t=%.2f x=%.2f z=%.2f rot=%.0f vz=%.2f calf_vmax=%.1f" % (row[0]*0.02+0.02, *row[1:]))

np.savez(sys.argv[2], **{k: np.array(v) for k, v in rec.items()})
print('ISO saved', sys.argv[2])

print('ISO  t     head_N   base_N  feet_in_contact  thigh/calf_N   rot   z')
for c in cont[40:80:2]: print('ISO %.2f  %7.0f  %7.0f  %d  %7.0f  %5.0f  %.2f' % (c[0]*0.02+0.02, *c[1:]))

print('ISO crouch_foot_slip_m (sum over feet, t<0.5s) =', round(slip_len, 2))
