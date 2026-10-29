"""Open-loop replay of IsaacLab actions in MuJoCo (Go2 Menagerie) to compare trajectories and find the dynamics gap."""
import math, sys, itertools
import mujoco, numpy as np

SCENE = "/home/ecarn/Documents/grasp_synth/Dexonomy/third_party/mujoco_menagerie/unitree_go2/scene.xml"
NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
DEFAULT = np.array([0, 0, 0, 0, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5])
VLIM = np.array([30.1] * 8 + [15.7] * 4)
ref = np.load(sys.argv[1]); acts = ref["act"]

def run(damping=None, floss=None, arm=None, vlim=True, kp=70., kd=3., foot_fric=None, z0=0.34, dt=0.005, mass_scale=1.0, tlim_scale=1.0):
    m = mujoco.MjModel.from_xml_path(SCENE); m.opt.timestep = dt
    if damping is not None: m.dof_damping[6:] = damping
    if floss is not None: m.dof_frictionloss[6:] = floss
    if arm is not None: m.dof_armature[6:] = arm
    if foot_fric is not None:
        for gn in ("FL", "FR", "RL", "RR"): m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gn), 0] = foot_fric
    m.body_mass[:] *= mass_scale
    d = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint") for n in NAMES]
    qa = np.array([m.jnt_qposadr[j] for j in jid]); va = np.array([m.jnt_dofadr[j] for j in jid])
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in NAMES]
    tl = np.array([m.actuator_ctrlrange[a, 1] for a in aid]) * tlim_scale
    d.qpos[:3] = [0, 0, z0]; d.qpos[3] = 1; d.qpos[qa] = DEFAULT; mujoco.mj_forward(m, d)
    out = {"z": [], "x": [], "rot": [], "q": []}; rot = 0.0
    for a in acts:
        tgt = DEFAULT + 3.0 * a
        for _ in range(int(round(0.02 / dt))):
            tq = kp * (tgt - d.qpos[qa]) - kd * d.qvel[va]
            d.ctrl[aid] = np.clip(tq, -tl, tl); mujoco.mj_step(m, d)
            if vlim: d.qvel[va] = np.clip(d.qvel[va], -VLIM, VLIM)
        rot += -d.qvel[4] * 0.02
        out["z"].append(d.qpos[2]); out["x"].append(d.qpos[0]); out["rot"].append(math.degrees(rot)); out["q"].append(d.qpos[qa].copy())
    return {k: np.array(v) for k, v in out.items()}

def report(name, o):
    idx = [24, 36, 49, 55, 70, 99]  # t = 0.5, 0.74, 1.0, 1.1, 1.4, 2.0
    q_rmse = np.sqrt(((o["q"][:55] - ref["qpos"][:55]) ** 2).mean())
    print(f"{name:34s} qRMSE(0-1.1s)={q_rmse:.3f} | " + " ".join(f"t={(i+1)*.02:.2f}: z={o['z'][i]:.2f} x={o['x'][i]:+.2f} rot={o['rot'][i]:4.0f}" for i in idx[:4]) + f" | zmax={o['z'].max():.2f}", flush=True)

print("ISAAC (reference)                  qRMSE=0     | " + " ".join(f"t={(i+1)*.02:.2f}: z={ref['z'][i]:.2f} x={ref['x'][i]:+.2f} rot={ref['rot'][i]:4.0f}" for i in [24, 36, 49, 55]) + f" | zmax={ref['z'].max():.2f}")
cfgs = [
  ("menagerie (d2,fl.2,arm.01)", {}),
  ("no fl/damp, arm0.01", dict(damping=0, floss=0, arm=0.01)),
  ("no fl/damp, arm0", dict(damping=0, floss=0, arm=0.0)),
  ("damp0 fl0.2 arm.01", dict(damping=0)),
  ("damp2 fl0 arm.01", dict(floss=0)),
  ("menagerie, no vlim", dict(vlim=False)),
  ("menagerie, mass x0.9", dict(mass_scale=0.9)),
  ("menagerie, dt 0.002", dict(dt=0.002)),
]
for n, kw in cfgs: report(n, run(**kw))

# ---- crouch phase only (first 0.5 s): per-joint position / target comparison
print("\nCROUCH PHASE (t=0.2..0.5s), joint order:", NAMES)
tgt = DEFAULT + 3.0 * acts
for i in (9, 14, 19, 24):
    o = run(); 
    print(f"t={(i+1)*.02:.2f} target  ", np.round(tgt[i], 2).tolist())
    print(f"       isaac q  ", np.round(ref['qpos'][i], 2).tolist())
    print(f"       mujoco q ", np.round(o['q'][i], 2).tolist(), f" z isaac={ref['z'][i]:.2f} mj={o['z'][i]:.2f}")
print("max |act| per joint over 0-0.5s", np.round(np.abs(acts[:25]).max(0), 2).tolist())
print("Isaac torque at t=0.3 s", np.round(ref['tq'][14], 1).tolist())
