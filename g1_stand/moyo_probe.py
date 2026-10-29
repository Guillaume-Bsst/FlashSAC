import numpy as np, torch, smplx, sys, glob
ROOT = "/home/ecarn/Documents/wbt_rl/data/00_raw_datasets/models/models_smplx_v1_1/models"
J = ["pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2","L_ankle","R_ankle","spine3","L_foot","R_foot","neck","L_collar","R_collar","head","L_shoulder","R_shoulder","L_elbow","R_elbow","L_wrist","R_wrist"]
def joints(p, step=30):
    z = np.load(p, allow_pickle=True); idx = np.arange(0, len(z["poses"]), step)
    po = torch.tensor(z["poses"][idx], dtype=torch.float32); n = len(idx)
    m = smplx.create(ROOT, model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=16, batch_size=n)
    out = m(global_orient=po[:, :3], body_pose=po[:, 3:66], jaw_pose=po[:, 66:69], leye_pose=po[:, 69:72], reye_pose=po[:, 72:75],
            left_hand_pose=po[:, 75:120], right_hand_pose=po[:, 120:165], betas=torch.tensor(z["betas"][:16], dtype=torch.float32)[None].repeat(n, 1),
            transl=torch.tensor(z["trans"][idx], dtype=torch.float32), return_verts=False)
    return idx, out.joints[:, :22].detach().numpy(), float(z["mocap_frame_rate"])
if __name__ == "__main__":
    for p in sorted(glob.glob("/home/ecarn/moyo_data/YOGI_2_latest_smplx_neutral/*/*Pincha*")):
        idx, j, fps = joints(p); print(p.split("/")[-2], p.split("_")[-3:-1], "frames", idx[-1] + 30, "fps", fps)
        for k in range(0, len(idx), 4):
            pel, head, ank, elb, wr = j[k, 0], j[k, 15], j[k, [7, 8]].mean(0), j[k, [18, 19]].mean(0), j[k, [20, 21]].mean(0)
            print(f"  t={idx[k]/fps:5.1f}s pelvis_z={pel[2]:.2f} head_z={head[2]:.2f} ankle_z={ank[2]:.2f} elbow_z={elb[2]:.2f} wrist_z={wr[2]:.2f} min_z={j[k,:,2].min():.2f}")
