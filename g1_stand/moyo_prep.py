"""MOYO (AMASS SMPL-X neutral) clip -> holosoma retargeting input: global_joint_positions (T,22,3) at 30 fps + human height."""
import numpy as np, torch, smplx, sys, os
ROOT = "/home/ecarn/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models"
def run(path, out, dest_fps=30):
    z = np.load(path, allow_pickle=True); fps = float(z["mocap_frame_rate"]); idx = np.arange(0, len(z["poses"]), int(round(fps / dest_fps)))
    po = torch.tensor(z["poses"][idx], dtype=torch.float32); n = len(idx)
    m = smplx.create(ROOT, model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=16, batch_size=n)
    kw = dict(global_orient=po[:, :3], body_pose=po[:, 3:66], jaw_pose=po[:, 66:69], leye_pose=po[:, 69:72], reye_pose=po[:, 72:75],
              left_hand_pose=po[:, 75:120], right_hand_pose=po[:, 120:165], betas=torch.tensor(z["betas"][:16], dtype=torch.float32)[None].repeat(n, 1),
              transl=torch.tensor(z["trans"][idx], dtype=torch.float32))
    with torch.no_grad():
        o = m(return_verts=True, **kw)
    j = o.joints[:, :22].numpy(); v0 = o.vertices[0].numpy()
    height = float(v0[:, 2].max() - v0[:, 2].min())
    np.savez(out, global_joint_positions=j.astype(np.float32), height=np.float32(height), fps=np.float32(dest_fps))
    print(os.path.basename(out), "frames", n, "height %.2f" % height, "pelvis_z0 %.2f" % j[0, 0, 2], flush=True)
if __name__ == "__main__":
    D = "/home/ecarn/moyo_data/YOGI_2_latest_smplx_neutral/train/"
    for k in "abcd":
        p = [f for f in os.listdir(D) if f"Pincha_Mayurasana_-{k}_" in f][0]
        run(D + p, f"/home/ecarn/moyo_rt/pincha_{k}.npz")
