"""Sim2sim: run a G1 forearm-stand FlashSAC checkpoint (trained in IsaacLab) in MuJoCo, same obs / action / PD as the
IsaacLab env, and save the trajectory in g1_record's npz format (render with g1_render.py).

--mode matched : MJCF made to match the IsaacLab setup (no joint damping / frictionloss, IsaacLab armature, no self-collision)
--mode raw     : MJCF as shipped (joint damping 0.05, frictionloss 0.1-0.2, self-collision on, full-mesh forearm collision)
Both keep the MJCF collision meshes, so the forearm (elbow_link) can touch the ground here, unlike the IsaacLab USD.
"""
import argparse, json, os, re
import numpy as np, torch, mujoco as mj, hydra, gymnasium as gym
from omegaconf import OmegaConf
from flash_rl.agents import create_agent

M = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"
HERE = os.path.dirname(os.path.abspath(__file__))
# IsaacLab joint order (ManagerBasedRLEnv articulation order), as saved by g1_record.py
JN = [str(n) for n in np.load(os.path.join(HERE, "g1_stand/traj_C15_0/C15_step14648.npz"))["joint_names"]]
POSE = json.load(open(os.path.join(HERE, "g1_stand/dolphin2_pose.json")))


def per_joint(table, default=None):
    out = []
    for n in JN:
        v = default
        for pat, val in table.items():
            if re.fullmatch(pat, n):
                v = val; break
        assert v is not None, n
        out.append(v)
    return np.array(out, dtype=np.float64)


SCALE_STD = {".*_hip_pitch_joint": 2.0, ".*_knee_joint": 2.0, ".*_shoulder_pitch_joint": 2.0, ".*_wrist_.*": 0.5,
             ".*_hip_(roll|yaw)_joint": 1.0, ".*_ankle_.*": 1.0, "waist_.*_joint": 1.0, ".*_shoulder_(roll|yaw)_joint": 1.0,
             ".*_elbow_joint": 1.0}
SCALE_WIDE = {".*_hip_pitch_joint": 3.5, ".*_knee_joint": 2.0, ".*_shoulder_pitch_joint": 2.0, ".*_wrist_roll_joint": 3.0,
              ".*_wrist_pitch_joint": 0.5, ".*_wrist_yaw_joint": 1.5, ".*_hip_(roll|yaw)_joint": 1.0, ".*_ankle_.*": 1.5,
              "waist_.*_joint": 1.0, ".*_shoulder_(roll|yaw)_joint": 1.5, ".*_elbow_joint": 2.0}
KP = per_joint({".*_hip_.*": 100.0, ".*_knee_joint": 200.0, ".*_ankle_.*": 40.0, "waist_.*": 60.0,
                ".*_shoulder_.*": 60.0, ".*_elbow_joint": 60.0, ".*_wrist_roll_joint": 60.0, ".*_wrist_(pitch|yaw)_joint": 15.0})
KD = per_joint({".*_hip_.*": 2.5, ".*_knee_joint": 5.0, ".*_ankle_.*": 2.0, "waist_.*": 3.0,
                ".*_shoulder_.*": 3.0, ".*_elbow_joint": 3.0, ".*_wrist_roll_joint": 3.0, ".*_wrist_(pitch|yaw)_joint": 1.0})
TMAX = per_joint({".*_hip_.*": 88.0, ".*_knee_joint": 139.0, ".*_ankle_.*": 50.0, "waist_yaw_joint": 88.0, "waist_.*": 50.0,
                  ".*_shoulder_.*": 25.0, ".*_elbow_joint": 25.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_(pitch|yaw)_joint": 5.0})
ARM = per_joint({".*_hip_.*": 0.03, ".*_knee_joint": 0.03, ".*_ankle_.*": 0.03, "waist_.*": 0.01, ".*_shoulder_.*": 0.01,
                 ".*_elbow_joint": 0.01, ".*_wrist_roll_joint": 0.01, ".*_wrist_(pitch|yaw)_joint": 0.005})
Q0 = np.array([POSE["joint_pos"][n] for n in JN])


def build_model(mode):
    m = mj.MjModel.from_xml_path(M)
    m.opt.timestep = 0.005
    m.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    jid = [m.joint(n).id for n in JN]
    dof = np.array([m.jnt_dofadr[j] for j in jid])
    m.dof_armature[dof] = ARM
    if mode == "matched":
        m.dof_damping[dof] = 0.0
        m.dof_frictionloss[dof] = 0.0
        floor = m.geom("floor").id
        for g in range(m.ngeom):  # robot collides with the floor only (IsaacLab: enabled_self_collisions=False)
            if g != floor and m.geom_contype[g] != 0:
                m.geom_contype[g] = 2; m.geom_conaffinity[g] = 1
    return m, np.array([m.jnt_qposadr[j] for j in jid]), dof, np.array([m.actuator(n.replace("_joint", "")).id for n in JN])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--mode", default="matched", choices=["matched", "raw"]); p.add_argument("--wide", action="store_true")
    p.add_argument("--start", default="held", choices=["held", "stand"]); p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seconds", type=float, default=10.0); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path="./configs")
    ov = ["env=isaaclab", "env.env_name=FlashSAC-G1ForearmStand-v0", "agent=flashSAC", "agent.asymmetric_observation=false",
          "agent.buffer_max_length=1", "agent.buffer_min_length=1", "agent.buffer_device_type=cuda", "gamma=0.95", "n_step=1"]
    cfg = hydra.compose(config_name="flashSAC_base", overrides=ov); OmegaConf.resolve(cfg)
    nj = len(JN)
    agent = create_agent(observation_space=gym.spaces.Box(-np.inf, np.inf, (1, 3 + 3 + 3 * nj)),
                         action_space=gym.spaces.Box(-1.0, 1.0, (1, nj)), env_info={}, cfg=cfg.agent)
    agent.load(a.ckpt)
    scale = per_joint(SCALE_WIDE if a.wide else SCALE_STD)

    m, qadr, dof, act = build_model(a.mode); d = mj.MjData(m)
    B = np.load(os.path.join(HERE, "g1_stand/bank_a.npz")); bj = [str(n) for n in B["joint_names"]]; bidx = [bj.index(n) for n in JN]
    up = B["torso_up"]; pool = np.where(up < -0.85)[0] if a.start == "held" else np.where((up > 0.9) & (np.arange(len(up)) < 300))[0]
    rng = np.random.default_rng(a.seed)
    frames, epi = [], []
    for e in range(a.episodes):
        fr = rng.choice(pool)
        mj.mj_resetData(m, d)
        d.qpos[0:3] = B["root_pos"][fr]; d.qpos[2] += 0.01; d.qpos[3:7] = B["root_quat"][fr]
        d.qpos[qadr] = B["joint_pos"][fr][bidx]; d.qvel[0:3] = B["root_vel"][fr]; d.qvel[dof] = B["joint_vel"][fr][bidx]
        mj.mj_forward(m, d)
        x0 = d.qpos[0:2].copy()
        last = np.zeros(nj)
        snap = lambda: (frames.append(np.concatenate([[d.qpos[0] - x0[0], d.qpos[1] - x0[1], d.qpos[2]], d.qpos[3:7], d.qpos[qadr]])), epi.append(e))
        snap()
        for k in range(int(a.seconds / 0.02)):
            quat = d.qpos[3:7]; R = np.zeros(9); mj.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
            obs = np.concatenate([0.25 * d.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]), d.qpos[qadr] - Q0, 0.05 * d.qvel[dof], last])
            act_ = np.asarray(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs[None].astype(np.float32)}, training=False)).reshape(-1)
            last = np.clip(act_, -1.0, 1.0)
            tgt = Q0 + scale * last
            for _ in range(4):
                tau = np.clip(KP * (tgt - d.qpos[qadr]) - KD * d.qvel[dof], -TMAX, TMAX)
                d.ctrl[act] = tau
                mj.mj_step(m, d)
            snap()
    q = np.array(frames); np.savez(a.out, q=q, episode=np.array(epi), joint_names=np.array(JN), dt=0.02)
    # summary: time the pelvis stays above 0.4 m with the torso inverted
    for e in range(a.episodes):
        Q = q[np.array(epi) == e]
        zs = Q[:, 2]; ok = zs > 0.4
        bad = np.where(~ok[25:])[0]
        t = (bad[0] + 25) * 0.02 if len(bad) else a.seconds
        path = np.linalg.norm(np.diff(Q[25:int(t / 0.02), :2], axis=0), axis=-1).sum() if t > 0.6 else 0.0
        print(f"EP {e} start={a.start} mode={a.mode}: holds {t:.1f}s (pelvis>0.4m), pelvis path while held {path:.2f}m", flush=True)


if __name__ == "__main__":
    main()
