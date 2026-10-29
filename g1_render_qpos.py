"""Render a HoloV2 RunOutput (qpos (T,36): root xyz, quat xyzw, 29 joints in g1_29dof order) in MuJoCo to mp4 (side view, real time)."""
import sys, subprocess, numpy as np, mujoco as mj
M = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"
m = mj.MjModel.from_xml_path(M); d = mj.MjData(m)
W, H = 800, 600
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
ren = mj.Renderer(m, H, W)
cam = mj.MjvCamera(); cam.type = mj.mjtCamera.mjCAMERA_FREE; cam.distance = 3.0; import os; cam.azimuth = float(os.environ.get("CAMAZ", "90")); cam.elevation = -8
for path, out in zip(sys.argv[1::2], sys.argv[2::2]):
    q = np.load(path, allow_pickle=True)["qpos"]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", out], stdin=subprocess.PIPE)
    for i in range(len(q)):
        d.qpos[:] = q[i]; d.qpos[3:7] = q[i][[6, 3, 4, 5]]; mj.mj_forward(m, d)  # HoloV2 qpos stores the root quaternion as xyzw
        cam.lookat[:] = [d.qpos[0], d.qpos[1] * 0, 0.5]
        ren.update_scene(d, cam); p.stdin.write(ren.render().tobytes())
    p.stdin.close(); p.wait(); print("VIDEO", out, flush=True)
