"""Open-loop replay of the clip from RSI starts (no policy): how long does the sim follow the clip, how many actions clip."""
import os
os.environ["BF_PLAY"] = "1"
import numpy as np, torch
from flash_rl.envs.isaaclab import make_isaaclab_env
env = make_isaaclab_env(env_name="FlashSAC-G1ForearmStand-v0", num_envs=256, seed=0, headless=True)
import flash_rl.envs.g1_forearm_stand_isaaclab_env as G
u = env.envs.unwrapped; robot = u.scene["robot"]
env.reset(random_start_init=False)
B = u._g1s_bank
n = u.num_envs; T = 150
alive = torch.ones(n, dtype=torch.bool, device=u.device); follow = torch.zeros(n, device=u.device); following = torch.ones(n, dtype=torch.bool, device=u.device)
fr0 = u._g1s_fr.clone(); sat = []; jerr_log = []
for k in range(T):
    raw = G.demo_actions_raw(u); sat.append(float((raw.abs() > 1.0).float().mean()))
    a = G.demo_actions(u)
    _, _, term, trunc, _ = env.step(a.cpu().numpy())
    done = torch.as_tensor(np.logical_or(term, trunc), device=u.device)
    # clip state at current time
    f = (fr0.float() + (k + 1) * u.step_dt * G.CLIP_FPS).clamp(0, B["n"] - 1).long()
    jerr = (robot.data.joint_pos - B["jp"][f]).abs().mean(1)
    perr = (robot.data.root_pos_w - u.scene.env_origins - torch.cat([B["pos"][f][:, :2], B["pos"][f][:, 2:]], 1)).norm(dim=1)
    ok = (jerr < 0.15) & (perr < 0.15)
    following &= ok & alive & ~done
    follow += following.float() * u.step_dt
    alive &= ~done
    if k in (10, 25, 50, 100): jerr_log.append((k, float(jerr[alive].mean()) if alive.any() else -1, float(perr[alive].mean()) if alive.any() else -1))
held = B["held"]; is_held = torch.isin(fr0, held)
print(f"DEMO follow_s median={float(follow.median()):.2f} p25={float(follow.quantile(.25)):.2f} p75={float(follow.quantile(.75)):.2f} max={float(follow.max()):.2f}")
print(f"DEMO follow_s held-start median={float(follow[is_held].median()) if is_held.any() else -1:.2f}  other={float(follow[~is_held].median()):.2f}")
raw_all = []
print(f"DEMO frac actions saturated={np.mean(sat):.3f}  errors(k, joint_mae, root_err)={jerr_log}")
# which joints saturate: over all clip frames, fraction of frames where |raw action| > 1
term = u.action_manager._terms["joint_pos"]
raw = (B["jp"] - term._offset[:1]) / term._scale[:1]
fr = (raw.abs() > 1.0).float().mean(0)
for j in torch.argsort(fr, descending=True)[:10].tolist():
    print(f"DEMO sat {robot.joint_names[j]:28s} frac_frames={float(fr[j]):.2f} max|a|={float(raw[:, j].abs().max()):.2f} scale={float(term._scale[0, j]):.1f}")
held = B["held"]
frh = (raw[held].abs() > 1.0).float().mean(0)
print("DEMO sat in held frames:", [(robot.joint_names[j], round(float(frh[j]), 2)) for j in torch.argsort(frh, descending=True)[:6].tolist()])
