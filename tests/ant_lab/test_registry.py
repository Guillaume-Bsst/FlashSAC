import sys

import gymnasium


def test_register_idempotent_and_light():
    from ant_lab import registry

    registry.register()
    registry.register()
    assert registry.EXPERT_ID in gymnasium.registry
    assert registry.TRACK_ID in gymnasium.registry
    assert gymnasium.registry[registry.EXPERT_ID].entry_point == "ant_lab.expert_env:AntExpertEnv"
    assert gymnasium.registry[registry.TRACK_ID].kwargs["env_cfg_entry_point"] == "ant_lab.tracking_env:AntTrackingEnvCfg"
    assert "isaaclab" not in sys.modules
