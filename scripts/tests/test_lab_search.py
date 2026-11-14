"""python/tatbot_lab search: the archive of marks a study's read trials reached, the surrogate over its settings, and
batches aimed at what is not yet known, on a synthetic rig whose darkness and line width follow depth and speed."""
import numpy as np
import pytest
from tatbot_contracts import lab
from tatbot_lab import propose, search, space, store

CARTRIDGE, INK = "lutin-3rl-bugpin", "true_black"


@pytest.fixture()
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_LAB_ROOT", str(tmp_path / "lab"))
    record = {"schema": lab.STUDY, "id": "depth", "created": "2026-11-09T00:00:00Z", "question": "What lines?",
              "arm": "xarm", "stencil": {"pattern_id": "stencil-" + "ab" * 32, "layout": "silicone",
                                         "tile_mm": [111.0, 157.5]},
              "cell_mm": 16.0, "constants": {"battery_v": 6.0, "machine_hz": 51.0, "stroke_mm": 4.0},
              "cartridges": [CARTRIDGE], "inks": [INK], "palette": {INK: "inkcap_large_2", "water": "inkcap_large_1"},
              "techniques": list(space.TECHNIQUES),
              "bounds": {"trim_mm": [-3.0, -1.0], "speed_mm_s": [2.0, 8.0], "pitch_mm": [1.5, 1.5]}}
    store.create(record)
    return record


def world(settings: dict) -> dict:
    """The synthetic rig: deeper and slower is darker and wider, and deep and slow blows out."""
    depth, speed = -settings["trim_mm"], settings["speed_mm_s"]
    dark = 60.0 * (depth - 0.8) / speed
    width = 0.15 + 0.22 * depth / np.sqrt(speed)
    return {"dL_mean": round(dark, 2), "ink_width_mm": [round(width, 3)] * 5,
            "speckle": {"share": round(max(0.0, dark - 30.0) / 200.0, 3), "mean": 0.0}}


def drawn(study: dict, settings_list, metrics=world) -> dict:
    records = propose.add(study, [{"technique": "line", "cartridge": CARTRIDGE, "inks": [INK], "settings": s}
                                  for s in settings_list])
    store.append(study["id"], [{**r, "status": "read", "metrics": metrics(r["settings"])} for r in records])
    return store.trials(study["id"])


def test_every_technique_has_its_descriptors_and_bins_name_themselves():
    assert set(search.AXES) == set(space.TECHNIQUES)
    darkness = search.DARKNESS
    assert [darkness.bin(v) for v in (None, 1.0, 3.0, 12.0, 99.0)] == [None, 0, 1, 2, 5]
    assert [darkness.label(k) for k in (0, 2, 5)] == ["<3", "8-15", ">40"]


def test_the_surrogate_learns_a_smooth_response_and_says_what_it_would_have_predicted_without_each_point():
    rng = np.random.default_rng(1)
    x = rng.random((30, 2))
    y = 10 * x[:, 0] + 4 * np.sin(3 * x[:, 1])
    model = search.Surrogate(x, y)
    q = rng.random((50, 2))
    mean, sd = model.predict(q)
    assert np.sqrt(np.mean((mean - (10 * q[:, 0] + 4 * np.sin(3 * q[:, 1]))) ** 2)) < 0.6
    assert np.all(sd > 0)
    loo_mean, loo_sd = model.held_out()
    assert loo_mean[0] == pytest.approx(search.Surrogate(x[1:], y[1:]).predict(x[:1])[0][0])
    assert np.sqrt(np.mean((loo_mean - y) ** 2)) < 1.0 and np.all(loo_sd > 0)


def test_a_bins_elite_is_its_cleanest_trial_and_of_two_as_clean_the_faster(study):
    trials = drawn(study, [{"trim_mm": -2.0, "speed_mm_s": 4.0}, {"trim_mm": -2.0, "speed_mm_s": 4.5}])
    elites = search.archive(search.members(trials, "line", CARTRIDGE, INK))
    assert len(elites) == 1 and next(iter(elites.values()))["settings"]["speed_mm_s"] == 4.5


def test_with_nothing_read_a_batch_spreads_over_the_bounds_in_few_runs(study):
    proposals = search.batch(study, {}, "line", CARTRIDGE, INK, 8, max_runs=2)
    assert len(proposals) == 8 and proposals[0]["hypothesis"].startswith("novelty: as far")
    runs = {space.run_key({"cartridge": {"type": CARTRIDGE}, "settings": p["settings"]}) for p in proposals}
    assert len(runs) <= 2
    for p in proposals:
        assert -3.0 <= p["settings"]["trim_mm"] <= -1.0 and (p["settings"]["trim_mm"] / 0.25).is_integer()
        assert p["settings"]["pitch_mm"] == 1.5   # held at its bound
    records = propose.add(study, proposals)   # every one inside the study
    assert len(records) == 8


def test_novelty_aims_at_bins_no_trial_holds_and_reaches_new_marks(study):
    grid_settings = [{"trim_mm": t, "speed_mm_s": s} for t in (-1.5, -2.0) for s in (5.0, 6.5, 8.0)]
    trials = drawn(study, grid_settings)
    before = set(search.archive(search.members(trials, "line", CARTRIDGE, INK)))
    proposals = search.batch(study, trials, "line", CARTRIDGE, INK, 6, seed=3, max_runs=3)
    assert not any(p.get("repeat_of") for p in proposals)   # six clean trials: nothing to doubt yet
    assert any("a bin no trial holds" in p["hypothesis"] for p in proposals)
    reached = set(search.archive([{**p, "id": f"p{k}", "metrics": world(p["settings"])}
                                  for k, p in enumerate(proposals)]))
    assert reached - before   # the batch draws marks the study has not made yet


def test_a_trial_the_others_predict_badly_is_repeated(study):
    settings = [{"trim_mm": t, "speed_mm_s": s} for t in (-1.5, -2.0, -2.5) for s in (3.0, 5.0, 7.0)]
    odd = {"trim_mm": -2.0, "speed_mm_s": 5.0}

    def glitch(s):   # one cell read far off its neighbours: a smeared before frame, a bad dip
        m = world(s)
        return {**m, "dL_mean": m["dL_mean"] + 40.0} if s == odd else m

    trials = drawn(study, settings, glitch)
    proposals = search.batch(study, trials, "line", CARTRIDGE, INK, 5, max_runs=3)
    repeat = [p for p in proposals if p.get("repeat_of")]
    assert len(repeat) == 1 and repeat[0]["settings"] == odd and repeat[0]["hypothesis"].startswith("surprise")
    assert trials[repeat[0]["repeat_of"]]["settings"] == odd


def test_a_batch_whose_runs_are_all_it_may_vary_repeats_its_picks_for_their_floor(study):
    proposals = search.batch(study, {}, "line", CARTRIDGE, INK, 5, max_runs=2)
    assert len({tuple(sorted(p["settings"].items())) for p in proposals}) == 2
    assert sum(p["hypothesis"].startswith("repeat of this batch's pick") for p in proposals) == 3


def test_the_map_is_each_archives_elites_by_bin(study):
    trials = drawn(study, [{"trim_mm": -2.0, "speed_mm_s": 4.0}, {"trim_mm": -1.0, "speed_mm_s": 8.0}])
    [group] = search.groups(trials)
    table = search.grid(group)
    assert table["axes"] == ["darkness", "width_mm"] and table["read"] == 2 and len(table["elites"]) == 2
    assert table["bins"] == 36 and table["rows"][0] == "<3"


def test_a_search_needs_the_study_to_bound_a_depth_and_a_speed(study):
    with pytest.raises(ValueError, match="bounds no speed_mm_s"):
        search.batch({**study, "bounds": {"trim_mm": [-3.0, -1.0]}}, {}, "line", CARTRIDGE, INK, 2)
