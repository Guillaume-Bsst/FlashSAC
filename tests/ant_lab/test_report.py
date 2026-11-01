from ant_lab.metrics import EvalRow


def _run(algo, seed, success_at):
    rows = [EvalRow(algo, seed, "a", 600.0, 1000, 0.2, 0.5, 3.0)]
    if success_at is not None:
        rows.append(EvalRow(algo, seed, "b", float(success_at), 5000, 0.95, 0.05, 0.2))
    return rows


def test_summarize():
    from ant_lab.report_e1 import summarize

    runs = {"ppo/0": _run("ppo", 0, 7200), "ppo/1": _run("ppo", 1, 9000), "ppo/2": _run("ppo", 2, None),
            "fs/0": _run("flashsac", 0, 1800), "fs/1": _run("flashsac", 1, 2400), "fs/2": _run("flashsac", 2, 3000)}
    s = summarize(runs)
    assert s["median_s"]["ppo"] == 9000
    assert s["median_s"]["flashsac"] == 2400
    assert s["speedup"] == 3.75
    assert s["per_run"]["ppo/2"]["time_to_success_s"] is None
    assert s["per_run"]["fs/0"]["steps_to_success"] == 5000 and s["per_run"]["fs/0"]["algo"] == "flashsac"

    runs["ppo/1"] = _run("ppo", 1, None)
    s = summarize(runs)
    assert s["median_s"]["ppo"] is None and s["speedup"] is None
