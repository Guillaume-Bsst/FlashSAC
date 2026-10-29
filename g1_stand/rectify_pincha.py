"""Rectify a HoloV2 G1 retarget of a MOYO Pincha clip: during the hold the robot's pelvis is ~0.2 m behind the forearm support and its
torso leans ~30 deg, while the human stacks pelvis / shoulders over the elbows (~8 deg). Per-frame IK keeping the forearm contact points
(elbow / wrist_roll positions) at the retargeted values, pulling the pelvis over the elbow midpoint and the shoulder-pelvis axis to
vertical, staying close to the retargeted joints and smooth in time. Blend weight follows the human ankle height (0 standing -> 1 hold)."""
import sys, numpy as np, mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
src, human, out = sys.argv[1], sys.argv[2], sys.argv[3]
TILT = np.radians(float(sys.argv[4])) if len(sys.argv) > 4 else np.radians(8.0)
z = dict(np.load(src, allow_pickle=True)); Q = z["qpos"].copy()
H = np.load(human)["global_joint_positions"]
m = mj.MjModel.from_xml_path("/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"); d = mj.MjData(m)
bn = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
FORE = [bn(n) for n in ("left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link")]
SH = [bn("left_shoulder_pitch_link"), bn("right_shoulder_pitch_link")]; EL = [bn("left_elbow_link"), bn("right_elbow_link")]; PEL = bn("pelvis")
lo = m.jnt_range[1:, 0].copy(); hi = m.jnt_range[1:, 1].copy()
T = min(len(Q), len(H)); ank = H[:T, [7, 8]].mean(1)[:, 2]
b = np.clip((ank - 0.5) / 0.5, 0, 1); b = b * b * (3 - 2 * b)   # smoothstep on human ankle height
def to_mj(q):  # HoloV2 xyzw -> MuJoCo wxyz
    o = q.copy(); o[3:7] = q[[6, 3, 4, 5]]; return o
def from_mj(q):
    o = q.copy(); o[3:7] = q[[4, 5, 6, 3]]; return o
def build(ref, x):
    q = ref.copy(); q[0:3] += x[0:3]
    r = (R.from_rotvec(x[3:6]) * R.from_quat(ref[[4, 5, 6, 3]])).as_quat(); q[3:7] = [r[3], r[0], r[1], r[2]]
    q[7:] = np.clip(ref[7:] + x[6:], lo, hi); return q
def fk(q):
    d.qpos[:] = q; mj.mj_forward(m, d); return d.xpos
def metrics(q):
    X = fk(q); sh = X[SH].mean(0); pel = X[PEL]; v = (sh - pel) / np.linalg.norm(sh - pel)
    return np.degrees(np.arccos(np.clip(-v[2], -1, 1))), (pel - X[EL].mean(0))[:2]
Qo = Q.copy(); xprev = np.zeros(35)
for t in range(T):
    if b[t] < 0.01: continue
    ref = to_mj(Q[t]); X0 = fk(ref); fore0 = X0[FORE].copy(); off0 = (X0[PEL] - X0[EL].mean(0))[:2]
    def res(x):
        q = build(ref, x); X = fk(q)
        r = [(X[FORE] - fore0).ravel() * 30.0]                                            # keep forearm contact points
        off = (X[PEL] - X[EL].mean(0))[:2]; r.append((off - (1 - b[t]) * off0) * 12.0 * b[t])     # pelvis over the elbow midpoint
        v = (X[SH].mean(0) - X[PEL]); v = v / np.linalg.norm(v); r.append([(v[2] + np.cos(TILT)) * 10.0 * b[t]])  # shoulder-pelvis axis ~vertical (inverted)
        r.append(x[6:] * 0.3); r.append(x[0:6] * 0.2)                                      # stay close to the retarget
        r.append((x - xprev) * 1.0)                                                          # temporal smoothness
        return np.concatenate([np.ravel(a) for a in r])
    s = least_squares(res, xprev, max_nfev=25)
    xprev = s.x; Q[t] = from_mj(build(ref, s.x))
    if t % 100 == 0: print(f"t={t/30:5.1f}s b={b[t]:.2f} cost={s.cost:.3f}", flush=True)
z["qpos"] = Q; np.savez(out, **z)
for t in range(354, 820, 100):
    tb, ob = metrics(to_mj(Qo[t])); ta, oa = metrics(to_mj(Q[t])); X = fk(to_mj(Q[t]))
    print(f"t={t/30:5.1f}s torso_tilt {tb:5.1f} -> {ta:5.1f} deg | pelvis-elbow {np.round(ob,2)} -> {np.round(oa,2)} | min_body_z {X[1:,2].min():.3f} | forearm contact shift {np.abs(X[FORE]-fk(to_mj(Qo[t]))[FORE]).max()*1000:.1f} mm")
print("max joint-limit proximity (rad):", float(np.minimum(Q[:T,7:]-lo, hi-Q[:T,7:]).min()))
