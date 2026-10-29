"""Run a G1 forearm-stand checkpoint in IsaacLab (1 env) and save the state trajectory (root pose + joints) to npz."""
import os
os.environ["BF_PLAY"] = "1"
import argparse
import hydra, numpy as np, torch
from omegaconf import OmegaConf
from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env
p = argparse.ArgumentParser(); p.add_argument("--checkpoints", nargs="+"); p.add_argument("--out_dir"); p.add_argument("--episodes", type=int, default=2); p.add_argument("--overrides", action="append", default=[])
a = p.parse_args()
OmegaConf.register_new_resolver("eval", lambda s: eval(s))
hydra.initialize(version_base=None, config_path="./configs")
cfg = hydra.compose(config_name="flashSAC_base", overrides=a.overrides); OmegaConf.resolve(cfg)
env = make_isaaclab_env(env_name=cfg.env.env_name, num_envs=1, seed=0, headless=True)
u = env.envs.unwrapped; r = u.scene["robot"]
_, info = env.reset(random_start_init=False)
agent = create_agent(observation_space=env.observation_space, action_space=env.action_space, env_info=info, cfg=cfg.agent)
os.makedirs(a.out_dir, exist_ok=True)
for ck in a.checkpoints:
    agent.load(ck); name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ck)))) + "_" + os.path.basename(ck)
    frames = []; epi = []
    for e in range(a.episodes):
        obs, _ = env.reset(random_start_init=False)
        d = r.data; org = u.scene.env_origins[0]
        def snap(): frames.append(np.concatenate([(d.root_pos_w[0] - org).cpu().numpy(), d.root_quat_w[0].cpu().numpy(), d.joint_pos[0].cpu().numpy()])); epi.append(e)
        snap()
        for k in range(int(u.max_episode_length)):
            act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False))
            obs, rew, term, trunc, _ = env.step(act); snap()
            if bool(term[0]) or bool(trunc[0]): break
    np.savez(os.path.join(a.out_dir, name + ".npz"), q=np.array(frames), episode=np.array(epi), joint_names=np.array(r.joint_names), dt=u.step_dt)
    print("SAVED", name, len(frames), flush=True)
