"""Env de tracking d'une référence cinématique sur Ant : recette BeyondMimic adaptée.

Configuration par variables d'environnement (lues à la création de la config) :
ANT_LAB_REF (obligatoire), ANT_LAB_K, ANT_LAB_ACTION_MODE (absolute | residual), ANT_LAB_RSI (adaptive | uniform | start).
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ant_lab import quat as Q
from ant_lab.config import (ACTION_SCALE_RAD, ENV_ACTION_MODE, ENV_K, ENV_REF, ENV_RSI, EVAL_INIT_NOISE, FPS,
                            K_DEFAULT, tracker_strength)
from ant_lab.expert_env import pd_robot_cfg
from ant_lab.reference import load_reference
from ant_lab.sampling import AdaptiveSampler
from ant_lab.tracking_math import (anchor_obs, heading_velocity, relative_body_targets, reward_terms,
                                   termination_flags, total_reward)


@configclass
class AntTrackingEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0  # remplacé par la durée du clip
    decimation = 2
    action_space = 8
    observation_space = 55
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average", restitution_combine_mode="average",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = pd_robot_cfg(1.0)
    reference_path: str = ""
    k: float = K_DEFAULT
    action_mode: str = "absolute"
    rsi: str = "adaptive"

    def __post_init__(self):
        if ENV_REF not in os.environ:
            raise RuntimeError(f"{ENV_REF} doit pointer vers une référence .npz (avec corps, voir fk_fill)")
        self.reference_path = os.environ[ENV_REF]
        self.k = float(os.environ.get(ENV_K, K_DEFAULT))
        self.action_mode = os.environ.get(ENV_ACTION_MODE, "absolute")
        self.rsi = os.environ.get(ENV_RSI, "adaptive")
        assert self.action_mode in ("absolute", "residual"), self.action_mode
        assert self.rsi in ("adaptive", "uniform", "start"), self.rsi
        with np.load(self.reference_path) as z:
            T = int(z["joint_pos"].shape[0])
        self.episode_length_s = (T - 1) / FPS
        self.robot = pd_robot_cfg(tracker_strength(self.k))


class AntTrackingEnv(DirectRLEnv):
    cfg: AntTrackingEnvCfg

    def __init__(self, cfg: AntTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        ref = load_reference(cfg.reference_path)
        assert ref.has_bodies, "la référence n'a pas de corps : passer par ant_lab.fk_fill"
        assert list(ref.joint_names) == list(self.robot.joint_names), (ref.joint_names, self.robot.joint_names)
        assert list(ref.body_names) == list(self.robot.body_names), (ref.body_names, self.robot.body_names)
        f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)  # noqa: E731
        self.ref = {k: f(getattr(ref, k)) for k in
                    ["root_pos", "root_quat", "joint_pos", "joint_vel", "root_lin_vel", "root_ang_vel",
                     "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"]}
        self.ref_T = ref.num_frames
        self._joint_ids, _ = self.robot.find_joints(".*")
        self.foot_ids = self.robot.find_bodies(".*_foot")[0]
        self.sampler = AdaptiveSampler(self.ref_T, FPS, device=str(self.device))
        N, J = self.num_envs, len(self._joint_ids)
        self.frame = torch.zeros(N, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(N, J, device=self.device)
        self.prev_actions = torch.zeros(N, J, device=self.device)
        self._err_sum = torch.zeros(N, device=self.device)
        self._vel_sum = torch.zeros(N, 2, device=self.device)
        self._ref_vel_sum = torch.zeros(N, 2, device=self.device)
        self._steps = torch.zeros(N, device=self.device)
        self.last_episode_stats: dict[str, torch.Tensor] = {}

    # --- scène -------------------------------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # --- actions -----------------------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        if self.cfg.action_mode == "residual":
            base = self.ref["joint_pos"][torch.clamp(self.frame + 1, max=self.ref_T - 1)]
        else:
            base = self.robot.data.default_joint_pos[:, self._joint_ids]
        self.robot.set_joint_position_target(base + ACTION_SCALE_RAD * self.actions, joint_ids=self._joint_ids)

    # --- état du robot et de la référence ----------------------------------------------------------------
    def _state(self):
        d = self.robot.data
        o = self.scene.env_origins
        fr = self.frame
        ref_anchor_pos, ref_anchor_quat = self.ref["body_pos"][fr, 0], self.ref["body_quat"][fr, 0]
        robot_anchor_pos, robot_anchor_quat = d.root_link_pos_w - o, d.root_link_quat_w
        pos_rel, quat_rel = relative_body_targets(ref_anchor_pos, ref_anchor_quat, self.ref["body_pos"][fr],
                                                  self.ref["body_quat"][fr], robot_anchor_pos, robot_anchor_quat)
        return dict(ref_anchor_pos=ref_anchor_pos, ref_anchor_quat=ref_anchor_quat,
                    robot_anchor_pos=robot_anchor_pos, robot_anchor_quat=robot_anchor_quat,
                    body_pos_rel=pos_rel, body_quat_rel=quat_rel,
                    robot_body_pos=d.body_link_pos_w - o[:, None], robot_body_quat=d.body_link_quat_w)

    def current_reward_terms(self) -> dict[str, torch.Tensor]:
        s = self._state()
        d = self.robot.data
        fr = self.frame
        return reward_terms(
            **s, ref_body_lin_vel=self.ref["body_lin_vel"][fr], ref_body_ang_vel=self.ref["body_ang_vel"][fr],
            robot_body_lin_vel=d.body_link_lin_vel_w, robot_body_ang_vel=d.body_link_ang_vel_w,
            actions=self.actions, prev_actions=self.prev_actions, joint_pos=d.joint_pos[:, self._joint_ids],
            joint_limits=d.joint_pos_limits[:, self._joint_ids])

    # --- fins d'épisode, reward, observations ------------------------------------------------------------
    def _get_dones(self):
        self.frame += 1
        s = self._state()
        terminated = termination_flags(
            ref_anchor_pos=s["ref_anchor_pos"], ref_anchor_quat=s["ref_anchor_quat"],
            robot_anchor_pos=s["robot_anchor_pos"], robot_anchor_quat=s["robot_anchor_quat"],
            body_pos_rel=s["body_pos_rel"], robot_body_pos=s["robot_body_pos"], foot_ids=self.foot_ids)
        time_out = self.frame >= self.ref_T - 1
        self.sampler.record_failures(self.frame[terminated & ~time_out])
        self.sampler.step()
        q = self.robot.data.joint_pos[:, self._joint_ids]
        self._err_sum += (q - self.ref["joint_pos"][self.frame]).abs().mean(-1)
        d = self.robot.data
        self._vel_sum += heading_velocity(d.root_lin_vel_w, d.root_link_quat_w)
        self._ref_vel_sum += heading_velocity(self.ref["root_lin_vel"][self.frame], self.ref["root_quat"][self.frame])
        self._steps += 1
        return terminated, time_out

    def _get_rewards(self):
        terms = self.current_reward_terms()
        self.extras["log"] = {f"reward/{k}": v.mean() for k, v in terms.items()}
        return total_reward(terms)

    def _get_observations(self) -> dict:
        d = self.robot.data
        fr = self.frame
        a = anchor_obs(self.ref["body_pos"][fr, 0], self.ref["body_quat"][fr, 0],
                       d.root_link_pos_w - self.scene.env_origins, d.root_link_quat_w)
        obs = torch.cat([
            self.ref["joint_pos"][fr], self.ref["joint_vel"][fr], a, d.root_lin_vel_b, d.root_ang_vel_b,
            d.joint_pos[:, self._joint_ids] - d.default_joint_pos[:, self._joint_ids],
            d.joint_vel[:, self._joint_ids], self.actions,
        ], dim=-1)
        return {"policy": obs}

    # --- resets ------------------------------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        done = env_ids[self._steps[env_ids] > 0]
        if len(done) > 0:
            self.last_episode_stats = {
                "env_ids": done.clone(),
                "completed": self.reset_time_outs[done].float(),
                "mean_joint_err": self._err_sum[done] / self._steps[done],
                # écart entre vitesses moyennes de l'épisode (repère de cap) : « court-il avec la référence ? »
                "mean_torso_vel_err": torch.linalg.norm(self._vel_sum[done] - self._ref_vel_sum[done], dim=-1)
                / self._steps[done],
                "length": self._steps[done].clone(),
            }
        super()._reset_idx(env_ids)
        n = len(env_ids)
        if self.cfg.rsi == "adaptive":
            frames = self.sampler.sample(n)
        elif self.cfg.rsi == "uniform":
            frames = torch.randint(0, self.ref_T - 1, (n,), device=self.device)
        else:
            frames = torch.zeros(n, dtype=torch.long, device=self.device)
        self.reset_to_frame(env_ids, frames)
        if self.cfg.rsi == "start":
            self._add_init_noise(env_ids)
        self._err_sum[env_ids] = 0.0
        self._vel_sum[env_ids] = 0.0
        self._ref_vel_sum[env_ids] = 0.0
        self._steps[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

    def reset_to_frame(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        frames = frames.to(self.device).long()
        o = self.scene.env_origins[env_ids]
        pose = torch.cat([self.ref["root_pos"][frames] + o, self.ref["root_quat"][frames]], dim=-1)
        vel = torch.cat([self.ref["root_lin_vel"][frames], self.ref["root_ang_vel"][frames]], dim=-1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids)
        self.robot.write_root_link_velocity_to_sim(vel, env_ids)
        self.robot.write_joint_state_to_sim(self.ref["joint_pos"][frames], self.ref["joint_vel"][frames],
                                            joint_ids=self._joint_ids, env_ids=env_ids)
        self.frame[env_ids] = frames

    def _add_init_noise(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        d = self.robot.data
        u = lambda *s: 2.0 * torch.rand(*s, device=self.device) - 1.0  # noqa: E731
        lim = d.joint_pos_limits[env_ids][:, self._joint_ids]
        q = torch.clamp(d.joint_pos[env_ids][:, self._joint_ids] + EVAL_INIT_NOISE["joint"] * u(n, len(self._joint_ids)),
                        lim[..., 0], lim[..., 1])
        self.robot.write_joint_state_to_sim(q, d.joint_vel[env_ids][:, self._joint_ids],
                                            joint_ids=self._joint_ids, env_ids=env_ids)
        pos = d.root_link_pos_w[env_ids].clone()
        pos[:, :2] += EVAL_INIT_NOISE["root_xy"] * u(n, 2)
        yaw = torch.zeros(n, 3, device=self.device)
        yaw[:, 2] = EVAL_INIT_NOISE["yaw"] * u(n)
        quat = Q.quat_mul(Q.quat_from_rotvec(yaw), d.root_link_quat_w[env_ids])
        self.robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
