"""Solve the target inverted pose (forearm stand / Pincha Mayurasana) with MuJoCo FK: both forearms flat and parallel on the ground,
upper arms vertical, torso and legs vertical, CoM above the forearm support."""
import mujoco as mj, numpy as np, json
from scipy.optimize import least_squares
M = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"
m = mj.MjModel.from_xml_path(M); d = mj.MjData(m)
bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
jx = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
qa = lambda n: m.jnt_qposadr[jx(n)]
JN = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
mass = m.body_mass.sum()
# x z pitch hip knee ankle waist sh_p sh_r sh_y elbow wr_roll wr_pitch
names = "x z pitch hip knee ankle waist sh_p sh_r elbow wr_roll".split()
def build(v):
    x, z, ph, hip, kn, an, wp, sp, sr, el, wr = v
    q = m.qpos0.copy(); q[0:3] = [x, 0, z]; q[3:7] = [np.cos(ph / 2), 0, np.sin(ph / 2), 0]
    for s, sg in (("left", 1), ("right", -1)):
        q[qa(f"{s}_hip_pitch_joint")] = hip; q[qa(f"{s}_knee_joint")] = kn; q[qa(f"{s}_ankle_pitch_joint")] = an
        q[qa(f"{s}_shoulder_pitch_joint")] = sp; q[qa(f"{s}_shoulder_roll_joint")] = sg * sr
        q[qa(f"{s}_elbow_joint")] = el; q[qa(f"{s}_wrist_roll_joint")] = sg * wr
    q[qa("waist_pitch_joint")] = wp
    return q
def res(v):
    q = build(v); d.qpos[:] = q; mj.mj_forward(m, d)
    P = lambda n: d.xpos[bid(n)]
    r = []
    for s in ("left", "right"):
        r.append((P(f"{s}_elbow_link")[2] - 0.045) * 30); r.append((P(f"{s}_wrist_roll_link")[2] - 0.045) * 30)
        r.append((P(f"{s}_wrist_roll_link")[0] - P(f"{s}_elbow_link")[0] + 0.10) * 10)  # forearm points toward body-forward (-x after the flip)
        sg = 1 if s == 'left' else -1
        r.append((sg * P(f"{s}_elbow_link")[1] - 0.17) * 10); r.append((P(f"{s}_wrist_roll_link")[1] - P(f"{s}_elbow_link")[1]) * 10)  # elbows shoulder-width, forearms parallel
        r.append((P(f"{s}_shoulder_pitch_link")[0] - P(f"{s}_elbow_link")[0]) * 10)      # upper arm vertical (shoulder above elbow)
    torso_z = d.xmat[bid("torso_link")].reshape(3, 3)[2, 2]; r.append((torso_z + 1) * 5)   # torso up-axis points down in the world
    r.append((np.mean([P(f"{s}_ankle_roll_link")[0] for s in ("left", "right")]) - P("pelvis")[0]) * 5)  # legs vertical
    com = (d.xipos * m.body_mass[:, None]).sum(0) / mass
    sx = np.mean([0.5 * (P(f"{s}_elbow_link")[0] + P(f"{s}_wrist_roll_link")[0]) for s in ("left", "right")])
    r.append((com[0] - sx) * 10)
    r += [v[3] * 1.0, v[4] * 1.0, v[6] * 1.0, v[5] * 0.5]  # straight body
    return np.array(r)
b_lo = [-1, 0.2, 2.9, -0.6, 0.0, -0.8, -0.4, -3.09, -0.6, -0.5, -1.5]
b_hi = [1, 2.0, 3.4, 0.6, 0.6, 0.8, 0.4, 2.6, 1.0, 2.09, 1.5]
best = None; rng = np.random.default_rng(0)
for k in range(60):
    v0 = np.array([rng.uniform(-.3, .3), rng.uniform(.8, 1.4), rng.uniform(2.95, 3.3), 0, 0, 0, 0, rng.uniform(-3, -1), 0, rng.uniform(0, 1.8), 0.0])
    s = least_squares(res, v0, bounds=(b_lo, b_hi))
    if best is None or s.cost < best.cost: best = s
v = best.x; q = build(v); d.qpos[:] = q; mj.mj_forward(m, d)
print("cost", round(best.cost, 4), dict(zip(names, v.round(3))))
P = lambda n: d.xpos[bid(n)].round(3)
print("pelvis", P("pelvis"), "shoulder", P("left_shoulder_pitch_link"), "elbow", P("left_elbow_link"), "wrist", P("left_wrist_roll_link"), "ankle", P("left_ankle_roll_link"), "head~torso", P("torso_link"))
com = (d.xipos * m.body_mass[:, None]).sum(0) / mass; print("com", com.round(3), "torso_up_z", round(d.xmat[bid("torso_link")].reshape(3, 3)[2, 2], 3))
ph = v[2]
json.dump({"root_pos": [float(v[0]), 0.0, float(v[1])], "root_quat_wxyz": [float(np.cos(ph/2)), 0, float(np.sin(ph/2)), 0],
           "joint_pos": {n: float(q[qa(n)]) for n in JN}}, open("pincha_pose.json", "w"), indent=1)
