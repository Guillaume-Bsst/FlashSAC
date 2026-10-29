"""Sim2sim of the IsaacLab Go2 backflip policy in SIMPLE (Pinocchio/Coal-based simulator, ~/simple-workspace).

Run with the pixi python (has `simple`, pinocchio, example_robot_data). The policy runs in a separate process
(policy_server.py, ~/.flashsac_env) and is queried over a pipe. Same control as in IsaacLab: 50 Hz, explicit PD
(kp 70, kd 3, torque limits 23.7/45.43), target = default + action_scale * a, joint velocity clamp 30.1/15.7 rad/s.
"""
import argparse, math, subprocess, sys, time
import numpy as np
import pinocchio as pin
import simple
sys.path.insert(0, "/home/ecarn/simple-workspace/simple-lab/sandbox/go2_inference")
from gym_simple_models import createGo2

PIN_NAMES = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
ISAAC_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
ISAAC_DEFAULT = np.array([0, 0, 0, 0, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5])
I2P = [ISAAC_NAMES.index(n) for n in PIN_NAMES]   # pinocchio index k -> isaac index
P2I = [PIN_NAMES.index(n) for n in ISAAC_NAMES]
KP, KD = 70.0, 3.0
TLIM = np.array([45.43 if "calf" in n else 23.7 for n in PIN_NAMES]); VLIM = np.array([15.7 if "calf" in n else 30.1 for n in PIN_NAMES])
DEF_P = ISAAC_DEFAULT[I2P]

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True); ap.add_argument("--action_scale", type=float, default=3.0)
ap.add_argument("--episodes", type=int, default=1); ap.add_argument("--joint_limit", action="store_true"); ap.add_argument("--damping", type=float, default=0.0)
ap.add_argument("--joint_friction", type=float, default=0.0); ap.add_argument("--no_vlim", action="store_true"); ap.add_argument("--quality", default="medium")
ap.add_argument("--material", default="metal"); ap.add_argument("--trace", action="store_true"); ap.add_argument("--viz", action="store_true"); ap.add_argument("--slowmo", type=float, default=3.0)
ap.add_argument("--zero_action", action="store_true"); ap.add_argument("--substeps", type=int, default=4); ap.add_argument("--z0", type=float, default=0.34)
args = ap.parse_args()

# ---- policy process
srv = subprocess.Popen(["/home/ecarn/.flashsac_env/bin/python", "/home/ecarn/FlashSAC/policy_server.py", "--checkpoint_path", args.checkpoint_path,
    "--overrides", "env=isaaclab", "--overrides", "env.env_name=FlashSAC-Backflip-Go2-v0", "--overrides", "agent=flashSAC", "--overrides", "agent.asymmetric_observation=false",
    "--overrides", "agent.buffer_max_length=1", "--overrides", "agent.buffer_min_length=1", "--overrides", "agent.buffer_device_type=cuda", "--overrides", "gamma=0.95", "--overrides", "n_step=1"],
    cwd="/home/ecarn/FlashSAC", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
assert srv.stdout.readline().strip() == b"READY"
def policy(obs): srv.stdin.write(obs.astype(np.float32).tobytes()); srv.stdin.flush(); return np.frombuffer(srv.stdout.read(48), dtype=np.float32).copy()

# ---- Simple engine
model, geom_model, visual_model = createGo2(use_collision_spheres_as_limits=False)
for attr in ("qref", "qinit", "offset", "scale"):
    if hasattr(model, attr): setattr(model, attr, None)
floor_id = geom_model.getGeometryId("floor")
have = {(p.first, p.second) for p in geom_model.collisionPairs}
for i, o in enumerate(geom_model.geometryObjects):  # createGo2 only collides feet with the floor; a backflip needs the whole body
    if i != floor_id and (i, floor_id) not in have and (floor_id, i) not in have:
        geom_model.addCollisionPair(pin.CollisionPair(i, floor_id))
print("collision pairs:", len(geom_model.collisionPairs), flush=True)
eng = simple.PhysicsEngine(model, geom_model, visual_model, simple.ConstraintSolverType.PIP, simple.DelassusOperatorType.RIGID_BODY)
nv = eng.model.nv
damp = np.zeros(nv); damp[6:] = args.damping; eng.setJointDamping(damp)
lb = np.zeros(nv); ub = np.zeros(nv); lb[6:] = -args.joint_friction; ub[6:] = args.joint_friction; eng.setJointFrictions(lb, ub)
lo = eng.model.lowerPositionLimit.copy(); hi = eng.model.upperPositionLimit.copy(); lo[:7] = -np.inf; hi[:7] = np.inf
if not args.joint_limit: lo[:] = -np.inf; hi[:] = np.inf
eng.setJointLimits(lo, hi)
eng.settings.warmstart_constraint_velocities = 1; eng.setMaxContactsPerCollisionPair(4)
eng.setSimulationQuality({"low": simple.SimulationQuality.LOW, "medium": simple.SimulationQuality.MEDIUM, "high": simple.SimulationQuality.HIGH}[args.quality])
s = eng.getConstraintSolverSettings(); s.solve_ncp = True; eng.setConstraintSolverSettings(s)
print("jointnames", [eng.model.names[i] for i in range(1, len(eng.model.names))][:4], "nq", eng.model.nq, "nv", nv, flush=True)

viz = None
if args.viz:
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    mv = MeshcatVisualizer(model, geom_model, visual_model); mv.initViewer(open=True); mv.loadViewerModel(); viz = mv

dt_sim = 0.02 / args.substeps
for ep in range(1, args.episodes + 1):
    q = pin.neutral(eng.model).copy(); q[:3] = [0, 0, args.z0]; q[3:7] = [0, 0, 0, 1]; q[7:] = DEF_P
    v = np.zeros(nv); a_prev = np.zeros(12)
    rot = 0.0; hmax = 0.0; xs = []; belly = 0; vpk = 0.0
    for k in range(100):
        R = pin.Quaternion(q[3:7]).toRotationMatrix()
        w_b = v[3:6]; g = R.T @ np.array([0, 0, -1.0])
        ph = math.pi * (k * 0.02) / 2
        obs = np.concatenate([w_b * 0.25, g, (q[7:] - DEF_P)[P2I], v[6:][P2I] * 0.05, a_prev,
                              [math.sin(ph), math.cos(ph), math.sin(ph / 2), math.cos(ph / 2), math.sin(ph / 4), math.cos(ph / 4)]])
        a = np.zeros(12) if args.zero_action else np.clip(policy(obs), -1, 1); a_prev = a
        tgt = DEF_P + args.action_scale * a[I2P]
        for _ in range(args.substeps):
            tq = np.clip(KP * (tgt - q[7:]) - KD * v[6:], -TLIM, TLIM)
            eng.step(q, v, np.concatenate([np.zeros(6), tq]), dt_sim, nsteps=1)
            q = eng.state.qnew.copy(); v = eng.state.vnew.copy()
            if not args.no_vlim: v[6:] = np.clip(v[6:], -VLIM, VLIM)
        rot += -v[4] * 0.02; hmax = max(hmax, q[2]); vpk = max(vpk, np.abs(v[6:]).max())
        if args.trace and k % 3 == 0: print(f"  t={(k+1)*0.02:.2f} z={q[2]:.2f} x={q[0]:+.2f} rot={math.degrees(rot):5.0f}", flush=True)
        if viz is not None: viz.display(q); time.sleep(0.02 * args.slowmo)
        if not np.all(np.isfinite(q)): print("NaN state"); break
    Rf = pin.Quaternion(q[3:7]).toRotationMatrix()
    print(f"Episode {ep}: pitch_rotation={-math.degrees(rot):.0f} deg max_height={hmax:.2f} m final_height={q[2]:.2f} m final_up_z={Rf[2,2]:.2f} base_x={q[0]:.2f} vmax={vpk:.1f}", flush=True)
srv.stdin.close()
