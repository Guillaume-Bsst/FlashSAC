import os, sys
os.environ["BF_PLAY"] = "1"
import numpy as np, torch
from flash_rl.envs.isaaclab import make_isaaclab_env
env = make_isaaclab_env(env_name="FlashSAC-G1ForearmStand-v0", num_envs=4, seed=0, headless=True)
u = env.envs.unwrapped; r = u.scene["robot"]; sens = u.scene.sensors["contact_forces"]
obs, info = env.reset(random_start_init=False)
print("OBS", obs.shape, "ACT", env.action_space.shape, "joints", r.num_joints, flush=True)
print("bodies", r.body_names, flush=True)
for k in range(150):
    obs, rew, term, trunc, _ = env.step(np.zeros((4, env.action_space.shape[-1]), dtype=np.float32))
    if k % 25 == 0 or k == 149:
        d = r.data; f = sens.data.net_forces_w.norm(dim=-1)[0]
        hit = [sens.body_names[i] for i in range(f.shape[0]) if f[i] > 1.0]
        print(f"k={k} pelvis_z={float(d.root_pos_w[0,2]):.3f} x={float(d.root_pos_w[0,0]):.3f} rew={float(rew[0]):.3f} term={bool(term[0])} contacts={hit}", flush=True)
print("SMOKE_DONE")
