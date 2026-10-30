import sys


def test_build_overrides():
    from ant_lab.train_flashsac import build_overrides

    o = build_overrides("AntLab-Expert-v0", "smoke", 0, 2048, 1_000_000, 2_000_000, 200, "x/ckpt")
    for item in ["agent.buffer_min_length=40960", "updates_per_interaction_step=2", "n_step=3",
                 "evaluation_per_interaction_step=null", "save_checkpoint_per_interaction_step=200",
                 "env.env_name=AntLab-Expert-v0", "group_name=ant_lab", "save_path=x/ckpt",
                 "agent.buffer_max_length=2000000", "num_train_envs=2048"]:
        assert item in o, item
    assert o[0] == "group_name=ant_lab" and o[-1] == "save_path=x/ckpt"
    assert "isaaclab" not in sys.modules
