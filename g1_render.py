"""Kinematic replay of an IsaacLab G1 trajectory (npz) in MuJoCo, rendered offscreen to mp4 (side view, half speed)."""
import sys, subprocess, numpy as np, mujoco as mj
M = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"
m = mj.MjModel.from_xml_path(M); d = mj.MjData(m)
W, H = 800, 600
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
ren = mj.Renderer(m, H, W)
qa = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i): m.jnt_qposadr[i] for i in range(1, m.njnt)}
cam = mj.MjvCamera(); cam.type = mj.mjtCamera.mjCAMERA_FREE; cam.distance = 2.6; cam.azimuth = 90; cam.elevation = -8
for path in sys.argv[1:]:
    z = np.load(path); q = z["q"]; names = [str(n) for n in z["joint_names"]]; ep = z["episode"]
    out = path.replace(".npz", ".mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", out], stdin=subprocess.PIPE)
    for i in range(len(q)):
        d.qpos[:] = m.qpos0; d.qpos[0:3] = q[i, 0:3]; d.qpos[3:7] = q[i, 3:7]
        for j, n in enumerate(names): d.qpos[qa[n]] = q[i, 7 + j]
        mj.mj_forward(m, d)
        cam.lookat[:] = [d.qpos[0], 0.0, 0.45]
        ren.update_scene(d, cam); p.stdin.write(ren.render().tobytes())
        if i + 1 < len(q) and ep[i + 1] != ep[i]:  # pause between episodes
            for _ in range(15): p.stdin.write(ren.render().tobytes())
    p.stdin.close(); p.wait(); print("VIDEO", out, flush=True)
