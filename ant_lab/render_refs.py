"""Rend les références cinématiques de l'Ant en MP4 (MuJoCo hors écran), superposées, caméra qui suit le torse.

Même géométrie que ant_lab.view_refs (poses de corps d'IsaacLab, capsules aux dimensions de l'Ant).
Bout de patte noir = pied au sol (critère E0). Écrit dans results/ant_lab/videos/.

usage : MUJOCO_GL=egl ~/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python -m ant_lab.render_refs \
            [ref ...] [--speed 0.25] [--seconds 3] [--name ...]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from ant_lab.config import RESULTS
from ant_lab.view_refs import COLORS, CONTACT_COLOR, LABELS, LIMB_RADIUS, TORSO_RADIUS, _align_z, _resolve, \
    load_clip, segment_endpoints

W, H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _mat_to_quat(R):
    import mujoco

    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def build_xml(clips, alphas):
    bodies = []
    for i, (k, c) in enumerate(clips.items()):
        r, g, b = (v / 255 for v in COLORS[i % len(COLORS)])
        a = alphas[i]
        rgba = f"{r:.3f} {g:.3f} {b:.3f} {a}"
        bodies.append(f'<body name="{k}_torso" mocap="true"><geom type="sphere" size="{TORSO_RADIUS}" '
                      f'rgba="{rgba}"/></body>')
        lengths = np.linalg.norm(np.subtract(*segment_endpoints(c, 0)[::-1]), axis=1)
        for j, L in enumerate(lengths):
            bodies.append(f'<body name="{k}_seg{j}" mocap="true"><geom type="capsule" '
                          f'size="{LIMB_RADIUS} {L / 2:.5f}" rgba="{rgba}"/></body>')
        for j in range(4):
            bodies.append(f'<body name="{k}_tip{j}" mocap="true"><geom name="{k}_tipg{j}" type="sphere" '
                          f'size="{LIMB_RADIUS * 1.08}" rgba="{rgba}"/></body>')
    return f"""<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.91" rgb2="0.80 0.80 0.78" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" height="32"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0.3 0.3 -1" castshadow="true"/>
    <geom type="plane" size="0 0 0.05" material="grid"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def render(clips, out, speed, seconds, alphas):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_string(build_xml(clips, alphas))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.2, 120.0, -18.0
    mocap = {model.body(i).name: model.body_mocapid[i] for i in range(model.nbody) if model.body_mocapid[i] >= 0}
    tipg = {k: [model.geom(f"{k}_tipg{j}").id for j in range(4)] for k in clips}
    base_rgba = {k: model.geom_rgba[tipg[k][0]].copy() for k in clips}
    black = np.array([*(v / 255 for v in CONTACT_COLOR), 1.0])
    font, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 22)

    fps = next(iter(clips.values()))["fps"]
    T = min(c["num_frames"] for c in clips.values())
    if seconds:
        T = min(T, int(round(seconds * fps)) + 1)
    lead = next(iter(clips.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", f"{fps * speed:g}", "-i", "-", "-vf", "fps=30",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                          stdin=subprocess.PIPE)
    for f in range(T):
        for k, c in clips.items():
            a, b = segment_endpoints(c, f)
            d = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
            R = _align_z(d)
            data.mocap_pos[mocap[f"{k}_torso"]] = c["body_pos"][f, 0]
            data.mocap_quat[mocap[f"{k}_torso"]] = c["body_quat"][f, 0]
            for j in range(len(a)):
                data.mocap_pos[mocap[f"{k}_seg{j}"]] = 0.5 * (a[j] + b[j])
                data.mocap_quat[mocap[f"{k}_seg{j}"]] = _mat_to_quat(R[j])
            for j in range(4):
                data.mocap_pos[mocap[f"{k}_tip{j}"]] = c["tips"][f, j]
                model.geom_rgba[tipg[k][j]] = black if c["contact"][f, j] else base_rgba[k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [lead["body_pos"][f, 0, 0], lead["body_pos"][f, 0, 1], 0.3]
        renderer.update_scene(data, cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.text((24, 18), f"t = {f / fps:5.2f} s   (×{speed:g})", fill=(40, 40, 40), font=font)
        y = 56
        for i, (k, c) in enumerate(clips.items()):
            feet = "".join("●" if on else "○" for on in c["contact"][f])
            draw.rectangle((24, y + 4, 42, y + 22), fill=COLORS[i % len(COLORS)])
            draw.text((52, y), f"{LABELS.get(k, k)}   pieds au sol {feet}", fill=(40, 40, 40), font=small)
            y += 32
        draw.text((24, H - 40), "bout de patte noir = pied au sol", fill=(90, 90, 90), font=small)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print("écrit :", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="*", default=["ref_s0", "ref_s1"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = tout le clip")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    model = json.loads((RESULTS / "ant_model.json").read_text())
    clips = {Path(r).stem: load_clip(_resolve(r), model) for r in args.refs}
    alphas = [1.0] + [0.55] * (len(clips) - 1)
    name = args.name or ("_".join(clips) + f"_x{args.speed:g}")
    render(clips, RESULTS / "videos" / f"{name}.mp4", args.speed, args.seconds, alphas)


if __name__ == "__main__":
    main()
