"""Ant expert : reward « avancer » d'Isaac-Ant-Direct-v0, mais commandé en PD et de force S_EXPERT."""
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.ant import ANT_CFG
from isaaclab_tasks.direct.ant.ant_env import AntEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from ant_lab.config import ACTION_SCALE_RAD, KD0, KP0, S_EXPERT, TAU0


def pd_robot_cfg(strength: float):
    """L'Ant d'IsaacLab avec des moteurs PD de force ``strength`` (Kp, Kd et couple max multipliés)."""
    return ANT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # PD explicite : IsaacLab calcule le couple à chaque pas de physique et l'envoie en effort. Le drive
        # implicite de PhysX n'agit pas sur les articulations de cet USD (couple « appliqué » saturé, rien ne bouge).
        actuators={"body": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=KP0 * strength,
                                              damping=KD0 * strength, effort_limit=TAU0 * strength,
                                              effort_limit_sim=TAU0 * strength)},
    )


@configclass
class AntExpertEnvCfg(AntEnvCfg):
    robot = pd_robot_cfg(S_EXPERT)


class AntExpertEnv(LocomotionEnv):
    cfg: AntExpertEnvCfg

    def _apply_action(self):
        default = self.robot.data.default_joint_pos[:, self._joint_dof_idx]
        target = default + ACTION_SCALE_RAD * torch.clamp(self.actions, -1.0, 1.0)
        self.robot.set_joint_position_target(target, joint_ids=self._joint_dof_idx)
