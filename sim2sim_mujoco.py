"""Sim2sim: run a FlashSAC go2-backflip policy (trained in Genesis) in MuJoCo (menagerie Go2).

Reproduces the Genesis control loop: 50 Hz policy, decimation 4, explicit PD (kp=70, kd=3) on torques,
1-step action latency, target = default + 0.5 * clip(6 * a, +-6), torque limits from the model.
Usage:
  python sim2sim_mujoco.py --checkpoint_path models/.../stepN [--headless] --overrides ... (same as play_genesis.py)
"""
import argparse, math, random, time
import hydra, mujoco, numpy as np, torch, gymnasium as gym
from omegaconf import OmegaConf
from flash_rl.agents import create_agent

SCENE = "/home/ecarn/Documents/grasp_synth/Dexonomy/third_party/mujoco_menagerie/unitree_go2/scene.xml"
DOF_NAMES = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
             "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]  # Genesis order
DEFAULT = np.array([0, .8, -1.5, 0, .8, -1.5, 0, 1., -1.5, 0, 1., -1.5])
ISAAC_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
               "FL_calf", "FR_calf", "RL_calf", "RR_calf"]  # IsaacLab breadth-first joint order
ISAAC_DEFAULT = np.array([0, 0, 0, 0, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5])
KP, KD, SCALE, RANGE, DECIM, DT_CTRL, T_EP = 70.0, 3.0, 0.5, 6.0, 4, 0.02, 2.0


def run(args):
    global DOF_NAMES, DEFAULT, SCALE, RANGE
    isaac = args.task == "isaac"
    args.vlim = isaac and not args.no_vlim
    if isaac:
        DOF_NAMES, DEFAULT, SCALE, RANGE = ISAAC_NAMES, ISAAC_DEFAULT, args.action_scale, 1.0
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path="./configs")
    cfg = hydra.compose(config_name="flashSAC_base", overrides=args.overrides)
    OmegaConf.resolve(cfg)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    agent = create_agent(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (48,) if isaac else (64,), np.float32),
        action_space=gym.spaces.Box(-1, 1, (12,), np.float32),
        env_info={"actor_observation_size": (48,), "asymmetric_obs": False} if isaac else {"non_privileged_obs_size": 60, "actor_observation_size": [60]},
        cfg=cfg.agent)
    agent.load(args.checkpoint_path)

    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = DT_CTRL / DECIM
    if args.foot_friction is not None:
        for gn in ("FL", "FR", "RL", "RR"): m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gn), 0] = args.foot_friction
        m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"), 0] = args.foot_friction
    if args.clean_joints:
        m.dof_damping[6:] = 0; m.dof_frictionloss[6:] = 0; m.dof_armature[6:] = args.armature
    d = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint") for n in DOF_NAMES]
    qadr = np.array([m.jnt_qposadr[j] for j in jid]); vadr = np.array([m.jnt_dofadr[j] for j in jid])
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in DOF_NAMES]
    aid = [a if a >= 0 else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_joint") for a, n in zip(aid, DOF_NAMES)]
    assert min(aid) >= 0, "actuator names"
    tlim = np.array([m.actuator_ctrlrange[a, 1] for a in aid])
    bad = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n + s) for n in ["FL", "FR", "RL", "RR"] for s in ["_thigh", "_calf"]}
    bad.add(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"))
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjv
        viewer = mjv.launch_passive(m, d)

    for ep in range(1, args.num_episodes + 1):
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0, 0, 0.36]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[qadr] = DEFAULT
        mujoco.mj_forward(m, d)
        a = np.zeros(12); last_a = np.zeros(12); vpk = np.zeros(12); slip_len = 0.0
        VLIM = np.array([15.7 if 'calf' in n else 30.1 for n in DOF_NAMES])
        ret_pitch, hmax, belly_steps, k = 0.0, 0.0, 0, 0
        for k in range(int(T_EP / DT_CTRL)):
            R = d.xmat[1].reshape(3, 3)
            w = d.qvel[3:6]; g = R.T @ np.array([0, 0, -1.0]); vb = R.T @ d.qvel[0:3]
            t = k * DT_CTRL
            ph = math.pi * t / 2
            phase6 = [math.sin(ph), math.cos(ph), math.sin(ph / 2), math.cos(ph / 2), math.sin(ph / 4), math.cos(ph / 4)]
            if isaac:  # [ang_vel*0.25, gravity, joint_pos_rel, joint_vel*0.05, last_action, phase]
                priv = np.concatenate([w * 0.25, g, d.qpos[qadr] - DEFAULT, d.qvel[vadr] * 0.05, a, phase6]).astype(np.float32)
            else:
                obs = np.concatenate([w * 0.25, g, d.qpos[qadr] - DEFAULT, d.qvel[vadr] * 0.05, a, last_a, phase6])
                priv = np.concatenate([obs, [d.qpos[2]], vb * 2.0]).astype(np.float32)
            act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": priv[None]}, training=False)).reshape(-1)
            if isaac:
                a = np.clip(act, -1.0, 1.0); exec_a = a  # no latency in IsaacLab
            else:
                last_a, a = a, np.clip(act * RANGE, -RANGE, RANGE)
                exec_a = last_a  # 1-step latency, as in Genesis
            target = DEFAULT + exec_a * SCALE
            for _ in range(DECIM):
                qd = d.qvel[vadr]
                tq = KP * (target - d.qpos[qadr]) - KD * qd
                d.ctrl[aid] = np.clip(tq, -tlim, tlim)
                vpk = np.maximum(vpk, np.abs(qd))
                mujoco.mj_step(m, d)
                if args.vlim:  # PhysX clamps joint velocities to the articulation max (30.1 hip/thigh, 15.7 calf)
                    d.qvel[vadr] = np.clip(d.qvel[vadr], -VLIM, VLIM)
            ret_pitch += d.qvel[4] * DT_CTRL; hmax = max(hmax, d.qpos[2])
            if k < 25:
                for gn in ("FL", "FR", "RL", "RR"):
                    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gn)
                    if any(gid in (c.geom1, c.geom2) and 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) for c in d.contact[:d.ncon]):
                        vv = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, vv, 0); slip_len += np.linalg.norm(vv[3:5]) * DT_CTRL
            if args.trace and k % 3 == 0:
                feet = sum(1 for c in d.contact[:d.ncon] if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and (m.geom_bodyid[c.geom1] if m.geom_bodyid[c.geom2] == 0 else m.geom_bodyid[c.geom2]) not in bad)
                bodyc = sum(1 for c in d.contact[:d.ncon] if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and (m.geom_bodyid[c.geom1] if m.geom_bodyid[c.geom2] == 0 else m.geom_bodyid[c.geom2]) in bad)
                slip = []
                for gn in ("FL", "FR", "RL", "RR"):
                    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gn)
                    inc = [c for c in d.contact[:d.ncon] if gid in (c.geom1, c.geom2) and 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
                    if inc:
                        v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, v, 0)
                        slip.append(f"{gn}:{np.linalg.norm(v[3:5]):.2f}")
                print(f"  base_x={d.qpos[0]:.2f} y={d.qpos[1]:.2f} foot_slip(m/s)={' '.join(slip) or '-'}")
                print(f"  t={k*DT_CTRL+DT_CTRL:.2f}s z={d.qpos[2]:.2f} rot={math.degrees(ret_pitch):7.0f} deg wy={d.qvel[4]:6.1f} rad/s vz={d.qvel[2]:5.2f} foot_ctc={feet} body_ctc={bodyc}", flush=True)
            if k > 55:  # after the flip: body/thigh/calf touching ground
                for c in d.contact[:d.ncon]:
                    if m.geom_bodyid[c.geom1] in bad or m.geom_bodyid[c.geom2] in bad:
                        if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                            belly_steps += 1; break
            if viewer is not None:
                if not viewer.is_running(): return
                viewer.sync(); time.sleep(DT_CTRL * args.slowmo)
        print(f"Episode {ep}: pitch_rotation={math.degrees(ret_pitch):.0f} deg max_height={hmax:.2f} m "
              f"final_height={d.qpos[2]:.2f} m final_up_z={d.xmat[1].reshape(3,3)[2,2]:.2f} "
              f"steps_with_body_contact_after_flip={belly_steps} base_x={d.qpos[0]:.2f} crouch_foot_slip_m={slip_len:.2f} calf_vpk={vpk[8:].max() if isaac else vpk[[2,5,8,11]].max():.1f}", flush=True)
        if viewer is not None: time.sleep(args.pause)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--overrides", action="append", default=[])
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--slowmo", type=float, default=3.0)
    p.add_argument("--pause", type=float, default=1.5)
    p.add_argument("--clean_joints", action="store_true", help="zero menagerie joint damping/frictionloss")
    p.add_argument("--armature", type=float, default=0.0)
    p.add_argument("--action_scale", type=float, default=3.0, help="isaac task: joint target = default + scale * a")
    p.add_argument("--task", choices=["genesis", "isaac"], default="genesis")
    p.add_argument("--no_vlim", action="store_true", help="disable the joint velocity limits (30.1 hip/thigh, 15.7 calf rad/s), on by default for --task isaac")
    p.add_argument("--vbrake", type=float, default=5.0)
    p.add_argument("--foot_friction", type=float, default=None, help="sliding friction coefficient of feet and floor (default: 0.8 / 1.0, max is used)")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--headless", action="store_true")
    run(p.parse_args())
