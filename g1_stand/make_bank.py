"""Reset-state bank (RSI) from the rectified MOYO retarget: root pose/vel + joint pos/vel by joint name."""
import json, sys, numpy as np, mujoco as mj
M = "/home/ecarn/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/scene_29dof_freebase.xml"
m = mj.MjModel.from_xml_path(M)
names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt) if m.jnt_type[i] != 0]
q = np.load(sys.argv[1])["qpos"]; dt = 1 / 30
pos = q[:, :3]; quat = q[:, [6, 3, 4, 5]]  # xyzw -> wxyz
jp = q[:, 7:]
sm = lambda x: np.gradient(x, dt, axis=0)
bank = dict(root_pos=pos, root_quat=quat, root_vel=sm(pos), joint_pos=jp, joint_vel=sm(jp), joint_names=np.array(names))
# torso_up z from body quaternion
m2 = m; d = mj.MjData(m)
up = []
for i in range(len(q)):
    d.qpos[:] = 0; d.qpos[:3] = pos[i]; d.qpos[3:7] = quat[i]; d.qpos[7:] = jp[i]; mj.mj_kinematics(m, d)
    up.append(d.xmat[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "torso_link")].reshape(3, 3)[2, 2])
bank["torso_up"] = np.array(up)
np.savez(sys.argv[2], **bank)
print(len(q), "frames; held (up<-0.85):", int((bank["torso_up"] < -0.85).sum()), "standing(up>0.9):", int((bank["torso_up"] > 0.9).sum()), names[:3])
