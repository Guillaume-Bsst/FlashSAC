"""Policy server: loads a FlashSAC checkpoint and answers obs -> action over stdin/stdout (binary float32).
Lets a process in another python env (e.g. the Simple pixi env) query the policy without importing flash_rl."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
out = sys.stdout.buffer; sys.stdout = sys.stderr  # keep stdout clean for the binary protocol
import argparse, numpy as np, gymnasium as gym, hydra, torch
from omegaconf import OmegaConf
from flash_rl.agents import create_agent

p = argparse.ArgumentParser(); p.add_argument("--checkpoint_path", required=True); p.add_argument("--obs_dim", type=int, default=48); p.add_argument("--act_dim", type=int, default=12)
p.add_argument("--overrides", action="append", default=[]); a = p.parse_args()
OmegaConf.register_new_resolver("eval", lambda s: eval(s))
hydra.initialize(version_base=None, config_path="./configs")
cfg = hydra.compose(config_name="flashSAC_base", overrides=a.overrides); OmegaConf.resolve(cfg)
agent = create_agent(observation_space=gym.spaces.Box(-np.inf, np.inf, (a.obs_dim,), np.float32), action_space=gym.spaces.Box(-1, 1, (a.act_dim,), np.float32),
                     env_info={"actor_observation_size": (a.obs_dim,), "asymmetric_obs": False}, cfg=cfg.agent)
agent.load(a.checkpoint_path)
out.write(b"READY\n"); out.flush()
nbytes = a.obs_dim * 4
while True:
    buf = sys.stdin.buffer.read(nbytes)
    if len(buf) < nbytes: break
    obs = np.frombuffer(buf, dtype=np.float32).reshape(1, -1)
    act = np.array(agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False)).reshape(-1).astype(np.float32)
    out.write(act.tobytes()); out.flush()
