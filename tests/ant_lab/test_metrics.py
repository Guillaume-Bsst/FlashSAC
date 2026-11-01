import os

from ant_lab.metrics import (
    EvalRow,
    is_success,
    list_checkpoints,
    read_rows,
    steps_to_success,
    time_to_success,
    write_rows,
)


def _row(elapsed, frac, err, steps=0, vel=0.1):
    return EvalRow("ppo", 0, "c", elapsed, steps, frac, err, vel)


def test_is_success_bounds():
    assert is_success(_row(0, 0.9, 0.099))
    assert not is_success(_row(0, 0.89, 0.05))
    assert not is_success(_row(0, 0.95, 0.1))


def test_success_requires_running_with_the_reference():
    # E0 : un robot qui agite les pattes sur place va au bout du clip avec une erreur articulaire faible
    assert is_success(_row(0, 1.0, 0.05, vel=0.49))
    assert not is_success(_row(0, 1.0, 0.05, vel=0.5))
    assert not is_success(_row(0, 1.0, 0.055, vel=4.16))


def test_time_to_success_order():
    rows = [_row(300, 0.95, 0.05, 30), _row(100, 0.5, 0.05, 10), _row(200, 0.92, 0.08, 20)]
    assert time_to_success(rows) == 200
    assert steps_to_success(rows) == 20
    assert time_to_success([_row(1, 0.1, 1.0)]) is None
    assert steps_to_success([_row(1, 0.1, 1.0)]) is None


def test_rows_roundtrip(tmp_path):
    p = tmp_path / "rows.csv"
    a = [EvalRow("flashsac", 1, "x/step10", 12.5, 20480, 0.75, 0.12, 0.8)]
    b = [EvalRow("ppo", 2, "model_5.pt", 30.0, 491520, 1.0, 0.03, 0.2)]
    write_rows(p, a)
    write_rows(p, b)
    assert read_rows(p) == a + b


def test_list_checkpoints(tmp_path):
    fs = tmp_path / "fs"
    (fs / "ckpt" / "step500").mkdir(parents=True)
    (fs / "ckpt" / "step100").mkdir(parents=True)
    for n, t in [(500, 2000), (100, 1000)]:
        f = fs / "ckpt" / f"step{n}" / "actor.pt"
        f.write_bytes(b"x")
        os.utime(f, (t, t))
    out = list_checkpoints(fs, "flashsac", 2048)
    assert [s for _, _, s in out] == [100 * 2048, 1_024_000]
    assert [m for _, m, _ in out] == [1000, 2000]

    ppo = tmp_path / "ppo"
    ppo.mkdir()
    for it, t in [(100, 50), (50, 10)]:
        f = ppo / f"model_{it}.pt"
        f.write_bytes(b"x")
        os.utime(f, (t, t))
    out = list_checkpoints(ppo, "ppo", 4096)
    assert [s for _, _, s in out] == [50 * 24 * 4096, 9_830_400]
    assert out[0][0].name == "model_50.pt"
