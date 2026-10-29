"""Replay a trained FlashSAC agent on a Genesis env (go2-backflip) with the Genesis viewer.

Usage (same Hydra overrides as training, so the agent architecture matches the checkpoint):
    DISPLAY=:1 python play_genesis.py --checkpoint_path models/.../stepN --num_episodes 6 --slowmo 3 \
        --overrides env=genesis --overrides env.env_name=go2-backflip --overrides agent=flashSAC \
        --overrides agent.asymmetric_observation=true --overrides agent.buffer_max_length=1 \
        --overrides agent.buffer_min_length=1
Prints the cumulative pitch rotation per episode (a full backflip is about +/-360 deg).
"""

import argparse
import math
import random
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.genesis import make_genesis_env
from flash_rl.envs.genesis_envs import go2_backflip


SHOW_VIEWER = True


def _get_env_with_viewer(num_envs: int, eval_mode: bool):  # replaces go2_backflip.get_env
    go2_backflip.gs.init(logging_level="warning")
    env_cfg, obs_cfg, reward_cfg, command_cfg = go2_backflip.get_cfgs()
    return go2_backflip.Go2BackflipEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=SHOW_VIEWER,
        eval=eval_mode,
        debug=False,
    )


def play(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    global SHOW_VIEWER
    SHOW_VIEWER = args.record is None
    go2_backflip.get_env = _get_env_with_viewer  # get_genesis_env imports it lazily
    env = make_genesis_env(env_name=cfg.env.env_name, num_envs=1, rescale_action=cfg.env.rescale_action, eval_mode=True)

    _, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)
    base = env.base_env
    dt = float(base.dt)

    obs, _ = env.reset()
    frames = []
    for ep in range(1, args.num_episodes + 1):
        ep_return, pitch_rot, max_height, steps, body_contact, twist = 0.0, 0.0, 0.0, 0, 0, 0.0
        done = False
        while not done:
            actions = np.array(
                agent.sample_actions(interaction_step=0, prev_transition={"next_observation": obs}, training=False)
            )
            obs, rew, term, trunc, _ = env.step(actions)
            ep_return += float(np.asarray(rew).reshape(-1)[0])
            pitch_rot += float(base.base_ang_vel[0, 1]) * dt
            max_height = max(max_height, float(base.base_pos[0, 2]))
            steps += 1
            yw = go2_backflip.gs_transform_by_quat(torch.tensor([[0.0, 1.0, 0.0]], device=base.device), base.base_quat[:1])
            twist = max(twist, float((yw - torch.tensor([[0.0, 1.0, 0.0]], device=base.device)).norm()))
            if steps > 55:
                f = base.link_contact_forces[0, base.penalized_contact_link_indices, :].norm(dim=-1)
                body_contact += int((f > 0.1).any())
            done = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
            if args.record is not None:
                p = base.base_pos[0].cpu().numpy()
                base._floating_camera.set_pose(pos=p + np.array([0.0, -2.2, 0.3]), lookat=p + np.array([0.0, 0.0, 0.0]))
                frames.append(np.asarray(base._floating_camera.render()[0]))
            else:
                time.sleep(dt * args.slowmo)
        print(
            f"Episode {ep}: return={ep_return:.2f} steps={steps} "
            f"pitch_rotation={math.degrees(pitch_rot):.0f} deg max_base_height={max_height:.2f} m "
            f"final_height={float(base.base_pos[0,2]):.2f} m final_up_z={-float(base.projected_gravity[0,2]):.2f} body_contact_steps_after_flip={body_contact} twist_max={twist:.2f}",
            flush=True,
        )
        time.sleep(args.pause)
    if args.record is not None:
        import imageio.v2 as imageio

        imageio.mimsave(args.record, frames, fps=int(round(1 / dt / args.slowmo)), macro_block_size=1, codec="libx264", pixelformat="yuv420p")
        print(f"saved {args.record} ({len(frames)} frames)", flush=True)
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default="./configs")
    p.add_argument("--config_name", type=str, default="flashSAC_base")
    p.add_argument("--overrides", action="append", default=[])
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=6)
    p.add_argument("--slowmo", type=float, default=3.0, help="sleep per step = dt * slowmo (viewer is real-time at 1)")
    p.add_argument("--record", type=str, default=None, help="write an mp4 here instead of opening the viewer")
    p.add_argument("--pause", type=float, default=1.0)
    play(p.parse_args())
