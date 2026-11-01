import json

import numpy as np

from ant_lab.config import RESULTS
from ant_lab.feasibility import foot_tip_centers
from ant_lab.view_refs import load_clip, segment_endpoints


def test_segments_join_bodies_and_end_on_foot_tips():
    model = json.loads((RESULTS / "ant_model.json").read_text())
    clip = load_clip(RESULTS / "refs" / "ref_s1.npz", model)
    a, b = segment_endpoints(clip, frame=123)          # (12, 3) chacun : 4 hanches, 4 cuisses, 4 tibias
    bp = clip["body_pos"][123]
    np.testing.assert_allclose(a[:4], bp[[0] * 4], atol=1e-6)          # les hanches partent du torse
    np.testing.assert_allclose(b[:4], bp[1:5], atol=2e-3)              # ... et arrivent sur les corps *_leg
    np.testing.assert_allclose(a[4:8], bp[1:5], atol=1e-6)
    np.testing.assert_allclose(b[4:8], bp[5:9], atol=2e-3)             # les cuisses arrivent sur les *_foot
    tips = foot_tip_centers(clip["body_pos"], clip["body_quat"], model["foot_ids"], model["foot_tip_offsets"])
    np.testing.assert_allclose(b[8:], tips[123], atol=1e-6)            # les tibias finissent aux bouts de pattes
