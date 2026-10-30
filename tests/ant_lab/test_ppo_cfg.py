import sys


def test_ppo_cfg():
    from ant_lab.ppo_cfg import train_cfg

    c = train_cfg(max_iterations=100, save_interval=37, seed=2, experiment="e1", run_name="seed2")
    assert c["algorithm"]["learning_rate"] == 1e-3
    assert c["algorithm"]["entropy_coef"] == 0.005
    assert c["num_steps_per_env"] == 24
    assert c["policy"]["actor_hidden_dims"] == [512, 256, 128]
    assert c["obs_groups"]["critic"] == ["policy"]
    assert c["save_interval"] == 37 and c["seed"] == 2 and c["max_iterations"] == 100
    # dict neuf à chaque appel : rsl_rl retire des clés
    assert train_cfg(1, 1, 0, "a", "b") is not train_cfg(1, 1, 0, "a", "b")
    assert "isaaclab" not in sys.modules
