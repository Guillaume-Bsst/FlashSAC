"""Visualise les références cinématiques de l'Ant dans viser (navigateur), superposées et rejouables.

Le robot est dessiné à partir des poses de corps enregistrées (cinématique directe d'IsaacLab) : torse en sphère,
hanches, cuisses et tibias en capsules aux dimensions de l'Ant. Bout de patte noir = pied au sol (même critère
que E0 : bas de la sphère à moins de 2 cm du sol).

viser est cassé dans l'env hssim, on utilise hsmujoco :
usage : ~/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python -m ant_lab.view_refs [ref ...] [--port 8080]
        (ref = nom dans results/ant_lab/refs, ou chemin .npz ; par défaut ref_s0 ref_s1)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ant_lab.config import RESULTS
from ant_lab.feasibility import contact_state, foot_tip_centers

LEG_IDS, FOOT_IDS = [1, 2, 3, 4], [5, 6, 7, 8]
TORSO_RADIUS, LIMB_RADIUS = 0.25, 0.08
LABELS = {"ref_s0": "σ = 0 (l'expert)", "ref_s1": "σ plancher (bruitée)"}
# palette catégorielle de référence (dataviz), ordre fixe
COLORS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]
CONTACT_COLOR = (25, 25, 25)


def _rot(q, v):
    """Rotation de v (..., 3) par q (..., 4) en wxyz."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _to_local(q, v):
    return _rot(q * np.array([1.0, -1, -1, -1]), v)


def load_clip(path, model) -> dict:
    d = np.load(path)
    if "body_pos" not in d.files:
        raise SystemExit(f"{path} n'a pas de poses de corps (passer par ant_lab.fk_fill)")
    bp, bq = d["body_pos"].astype(np.float64), d["body_quat"].astype(np.float64)
    tips = foot_tip_centers(bp, bq, model["foot_ids"], model["foot_tip_offsets"])
    _, contact = contact_state(tips, radius=model["foot_radius"])
    # offsets rigides (hanche dans le repère du torse, cuisse dans le repère de la jambe), médiane sur le clip
    hip = np.median(_to_local(bq[:, [0]], bp[:, LEG_IDS] - bp[:, [0]]), axis=0)
    thigh = np.median(_to_local(bq[:, LEG_IDS], bp[:, FOOT_IDS] - bp[:, LEG_IDS]), axis=0)
    return {"fps": float(d["fps"]), "body_pos": bp, "body_quat": bq, "tips": tips, "contact": contact,
            "hip_off": hip, "thigh_off": thigh, "num_frames": bp.shape[0]}


def segment_endpoints(clip, frame):
    """Extrémités (a, b) des 12 capsules : 4 hanches, 4 cuisses, 4 tibias, chacune (12, 3)."""
    p, q = clip["body_pos"][frame], clip["body_quat"][frame]
    torso = np.repeat(p[[0]], 4, 0)
    a = np.concatenate([torso, p[LEG_IDS], p[FOOT_IDS]])
    b = np.concatenate([torso + _rot(np.repeat(q[[0]], 4, 0), clip["hip_off"]),
                        p[LEG_IDS] + _rot(q[LEG_IDS], clip["thigh_off"]),
                        clip["tips"][frame]])
    return a, b


def _align_z(d):
    """Matrices (N, 3, 3) qui envoient +z sur les directions unitaires d (N, 3)."""
    z = np.array([0.0, 0, 1])
    out = np.empty((len(d), 3, 3))
    for i, di in enumerate(d):
        v, c = np.cross(z, di), float(z @ di)
        if np.linalg.norm(v) < 1e-9:
            out[i] = np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
            continue
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        out[i] = np.eye(3) + k + k @ k / (1 + c)
    return out


class Robot:
    """Un maillage par référence, mis à jour à chaque frame."""

    def __init__(self, scene, name, clip, color):
        import trimesh

        self.clip, self.color = clip, color
        a, b = segment_endpoints(clip, 0)
        lengths = np.linalg.norm(b - a, axis=1)
        self.caps = [trimesh.creation.capsule(height=L, radius=LIMB_RADIUS, count=[10, 10]) for L in lengths]
        for c in self.caps:
            c.vertices[:, 2] += 0.0  # capsule centrée sur l'origine, axe z
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=TORSO_RADIUS)
        self.sphere_v = sphere.vertices
        parts = [sphere] + self.caps
        offs = np.cumsum([0] + [len(m.vertices) for m in parts])
        faces = np.concatenate([m.faces + o for m, o in zip(parts, offs[:-1])])
        self.mesh = scene.add_mesh_simple(f"/world/{name}/body", self._vertices(0), faces, color=color,
                                          flat_shading=False)
        tip = trimesh.creation.icosphere(subdivisions=2, radius=LIMB_RADIUS * 1.08)
        self.tips = scene.add_batched_meshes_simple(
            f"/world/{name}/tips", tip.vertices, tip.faces, batched_wxyzs=np.tile([1.0, 0, 0, 0], (4, 1)),
            batched_positions=clip["tips"][0], batched_colors=self._tip_colors(0))
        path = clip["body_pos"][:, 0].copy()
        path[:, 2] = 0.005
        self.trail = scene.add_line_segments(f"/world/{name}/trail", np.stack([path[:-1], path[1:]], 1),
                                             colors=color, thickness=0.02)

    def _tip_colors(self, f):
        return np.array([CONTACT_COLOR if c else self.color for c in self.clip["contact"][f]], dtype=np.uint8)

    def _vertices(self, f):
        a, b = segment_endpoints(self.clip, f)
        d = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
        R = _align_z(d)
        mid = 0.5 * (a + b)
        vs = [self.sphere_v + self.clip["body_pos"][f, 0]]
        vs += [c.vertices @ R[i].T + mid[i] for i, c in enumerate(self.caps)]
        return np.concatenate(vs).astype(np.float32)

    def show(self, f, visible, trail):
        f = min(f, self.clip["num_frames"] - 1)
        self.mesh.visible = self.tips.visible = visible
        self.trail.visible = visible and trail
        if visible:
            self.mesh.vertices = self._vertices(f)
            self.tips.batched_positions = self.clip["tips"][f].astype(np.float32)
            self.tips.batched_colors = self._tip_colors(f)


def _resolve(ref):
    p = Path(ref)
    return p if p.suffix == ".npz" else RESULTS / "refs" / f"{ref}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="*", default=["ref_s0", "ref_s1"])
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    import viser

    model = json.loads((RESULTS / "ant_model.json").read_text())
    clips = {Path(r).stem: load_clip(_resolve(r), model) for r in args.refs}
    T = max(c["num_frames"] for c in clips.values())
    fps = next(iter(clips.values()))["fps"]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (1.8, -3.2, 1.6)
    server.initial_camera.look_at = (0.0, 0.0, 0.35)
    world = server.scene.add_frame("/world", show_axes=False)
    server.scene.add_grid("/world/grid", width=200, height=200, cell_size=0.25, section_size=1.0,
                          infinite_grid=True, plane_color=(250, 250, 248), plane_opacity=1.0)
    robots = {k: Robot(server.scene, k, c, COLORS[i % len(COLORS)]) for i, (k, c) in enumerate(clips.items())}

    legend = "  \n".join(f"<span style='color:rgb{COLORS[i % 4]}'>■</span> {LABELS.get(k, k)}"
                         for i, k in enumerate(clips))
    server.gui.add_markdown(f"**Références cinématiques de l'Ant**  \n{legend}  \n■ noir au bout d'une patte = pied au sol")
    shown = {k: server.gui.add_checkbox(LABELS.get(k, k), True) for k in clips}
    frame = server.gui.add_slider("Frame", 0, T - 1, 1, 0)
    playing = server.gui.add_checkbox("Lecture", True)
    speed = server.gui.add_dropdown("Vitesse", ("0.1", "0.25", "0.5", "1"), initial_value="0.25")
    follow = server.gui.add_checkbox("Suivre le torse", True)
    trail = server.gui.add_checkbox("Trace du torse", True)
    info = server.gui.add_markdown("")
    lead = next(iter(clips))

    pos, last, prev = 0.0, None, time.perf_counter()
    while True:
        now = time.perf_counter()
        dt, prev = now - prev, now
        if playing.value:
            pos = (pos + dt * fps * float(speed.value)) % T
            frame.value = int(pos)
        else:
            pos = float(frame.value)
        f = int(frame.value)
        state = (f, follow.value, trail.value, tuple(c.value for c in shown.values()))
        if state != last:
            last = state
            c0 = clips[lead]
            x, y = c0["body_pos"][min(f, c0["num_frames"] - 1), 0, :2]
            world.position = (-x, -y, 0.0) if follow.value else (0.0, 0.0, 0.0)
            for k, r in robots.items():
                r.show(f, shown[k].value, trail.value)
            lines = [f"t = {f / fps:.2f} s · torse x = {x:.2f} m"]
            for k, c in clips.items():
                g = min(f, c["num_frames"] - 1)
                feet = "".join("●" if on else "○" for on in c["contact"][g])
                lines.append(f"{LABELS.get(k, k)} : pieds {feet}, torse z = {c['body_pos'][g, 0, 2]:.3f} m")
            if len(clips) > 1:
                others = list(clips.values())[1:]
                gaps = [np.linalg.norm(c["body_pos"][min(f, c["num_frames"] - 1), 0] - c0["body_pos"][f, 0])
                        for c in others]
                lines.append("écart du torse avec " + LABELS.get(lead, lead) + " : "
                             + ", ".join(f"{100 * g:.1f} cm" for g in gaps))
            info.content = "  \n".join(lines)
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
