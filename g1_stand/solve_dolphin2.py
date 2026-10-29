"""Solve the 'dolphin' start pose (forearms + feet on the ground, hips high) with MuJoCo FK on the GR00T G1 29dof MJCF."""
import mujoco as mj, numpy as np, json, sys
from scipy.optimize import least_squares
M = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"
m = mj.MjModel.from_xml_path(M); d = mj.MjData(m)
bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
qa = lambda n: m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]
JN = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
lo = {n: m.jnt_range[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)][0] for n in JN}
hi = {n: m.jnt_range[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)][1] for n in JN}
com_mass = m.body_mass.sum()
# variables: x, z, pitch, hip, knee, ankle, waist_pitch, sh_pitch, sh_roll, elbow, wrist_roll
def build(v):
    x, z, ph, hip, kn, an, wp, sp, sr, el, wr = v
    q = m.qpos0.copy(); q[:] = m.qpos0
    q[0:3] = [x, 0, z]; q[3:7] = [np.cos(ph/2), 0, np.sin(ph/2), 0]
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
        r.append((P(f"{s}_ankle_roll_link")[2] - 0.036) * 20)
        R = d.xmat[bid(f"{s}_ankle_roll_link")].reshape(3, 3); r.append((R[2, 2] - 1) * 5)
        r.append((P(f"{s}_elbow_link")[2] - 0.045) * 20)
        r.append((P(f"{s}_wrist_roll_link")[2] - 0.045) * 20)
    com = (d.xipos * m.body_mass[:, None]).sum(0) / com_mass
    fx = np.mean([P(f"{s}_ankle_roll_link")[0] for s in ("left", "right")]); wx = np.mean([P(f"{s}_wrist_roll_link")[0] for s in ("left", "right")])
    r.append((com[0] - 0.5 * (fx + wx)) * 5)
    for sd, sg in (('left', 1), ('right', -1)):
        r.append((sg * P(f'{sd}_elbow_link')[1] - 0.17) * 10); r.append((P(f'{sd}_wrist_roll_link')[1] - P(f'{sd}_elbow_link')[1]) * 10)
    r.append((P('left_wrist_roll_link')[0] - P('left_elbow_link')[0] - 0.10) * 10)  # forearm points forward
    r.append(v[4] * 0.1)  # knees ~straight
    r.append((P("pelvis")[2] - 0.5) * 6)  # hips high
    r.append((fx - wx - 0.35) * 3)  # feet close to forearms (dolphin base)
    return np.array(r)
names = ["hip_pitch_joint", "knee_joint", "ankle_pitch_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint", "left_wrist_roll_joint"]
b_lo = [-1, 0.1, 0.3, -2.5, 0.0, -0.9, -0.5, -3.0, -0.6, 0.0, -1.5]
b_hi = [1, 1.0, 1.9, 0.5, 2.2, 0.5, 0.5, 2.6, 1.0, 2.09, 1.5]
best = None
rng = np.random.default_rng(0)
for k in range(40):
    v0 = np.array([rng.uniform(-.2, .2), rng.uniform(.4, .8), rng.uniform(0.5, 1.5), rng.uniform(-2, 0), rng.uniform(0, .4), rng.uniform(-.5, .5), rng.uniform(-.3, .3), rng.uniform(-2, 1), 0.1, rng.uniform(0, 1.5), 0])
    s = least_squares(res, v0, bounds=(b_lo, b_hi))
    if best is None or s.cost < best.cost: best = s
v = best.x; q = build(v); d.qpos[:] = q; mj.mj_forward(m, d)
print("cost", best.cost, "vars", dict(zip("x z pitch hip knee ankle waist sh_p sh_r elbow wrist_r".split(), v.round(3))))
P = lambda n: d.xpos[bid(n)].round(3)
print("pelvis", P("pelvis"), "torso", P("torso_link"), "elbow", P("left_elbow_link"), "wrist", P("left_wrist_roll_link"), "ankle", P("left_ankle_roll_link"), "knee", P("left_knee_link"))
com = (d.xipos * m.body_mass[:, None]).sum(0) / com_mass; print("com", com.round(3), "mass", com_mass)
ph = v[2]
out = {"root_pos": [float(v[0]), 0.0, float(v[1])], "root_quat_wxyz": [float(np.cos(ph/2)), 0, float(np.sin(ph/2)), 0],
       "joint_pos": {n: float(q[qa(n)]) for n in JN}}
json.dump(out, open("dolphin2_pose.json", "w"), indent=1)
