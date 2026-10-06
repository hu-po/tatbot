"""Job bookkeeping, with a fake generator.

Everything here is a **bookkeeping** test: no model runs, no image is drawn, no
GPU is touched. It proves the ledger survives crashes, resume, corruption,
duplicates and a second worker — not that any picture is any good.
"""
import json
import os

import pytest
from batch import (  # noqa: E402
    ConversionRefusedError,
    Job,
    JobBusyError,
    freeze_request,
    load_job,
    plan_items,
    replan,
    shard,
)
from contracts import GenerationSettings  # noqa: E402

SETTINGS = GenerationSettings(model="test/model", model_revision="a" * 40, steps=2, width=128, height=128)
SUBJECTS = ["a heron", "a fern", "a koi"]


def make_request(count=6, budget=0, subjects=SUBJECTS, settings=SETTINGS, seed=17):
    return freeze_request(plan_items(subjects, count, seed=seed), settings=settings, seed=seed,
                          backend="endpoint", replacement_budget=budget)


class FakeGenerator:
    """Deterministic bytes per request, with scripted failures and repeats."""

    def __init__(self, *, fail=(), refuse=(), repeat=None, crash_after=None):
        self.calls = []
        self.fail = set(fail)
        self.refuse = set(refuse)
        self.repeat = repeat or {}
        self.crash_after = crash_after

    def generate(self, request):
        self.calls.append(request)
        if self.crash_after is not None and len(self.calls) > self.crash_after:
            raise KeyboardInterrupt("simulated crash")
        if request.subject in self.fail:
            raise TimeoutError("generator did not answer")
        body = self.repeat.get(request.subject, f"{request.subject}:{request.seed}")
        png = b"\x89PNG\r\n\x1a\n" + body.encode()
        return png, {"seed": request.seed, "prompt": request.prompt, "seconds": 0.5,
                     "model": request.settings.model, "model_revision": request.settings.model_revision}

    def convert(self, png, meta, item):
        if item.subject in self.refuse:
            raise ConversionRefusedError("the model drew nothing usable")
        name = f"{item.ordinal:04d}-{item.key}"
        return {"artifacts": {f"{name}.svg": f"<svg data='{item.key}'/>", f"{name}.png": png},
                "record": {"id": item.key, "png_sha256": meta["png_sha256"]}}


def run(root, request, fake, **kwargs):
    with Job.open(root, request) as job:
        return job.run(generate=fake.generate, convert=fake.convert, **kwargs)


# ---- identity -------------------------------------------------------------
def test_item_keys_and_seeds_do_not_move_when_the_pool_changes():
    before = {item.key: item.seed for item in plan_items(["a heron", "a fern"], 4, seed=3)}
    after = {item.key: item.seed for item in plan_items(["a heron", "a fern", "a koi"], 6, seed=3)}
    assert before and all(after[key] == seed for key, seed in before.items())


def test_reordering_the_item_list_leaves_the_job_identical():
    items = plan_items(SUBJECTS, 6, seed=17)
    a = freeze_request(items, settings=SETTINGS, seed=17, backend="endpoint")
    b = freeze_request(list(reversed(items)), settings=SETTINGS, seed=17, backend="endpoint")
    assert a["job_id"] == b["job_id"]


def test_sharding_preserves_every_request_identity():
    items = plan_items(SUBJECTS, 9, seed=17)
    parts = [shard(items, index=i, of=3) for i in range(3)]
    assert sum(len(part) for part in parts) == 9
    rebuilt = {item.key: item.request(SETTINGS).digest for part in parts for item in part}
    assert rebuilt == {item.key: item.request(SETTINGS).digest for item in items}


def test_changed_model_settings_make_a_different_job():
    base = make_request()
    assert replan(base, settings=SETTINGS)["job_id"] == base["job_id"]
    other = replan(base, settings=SETTINGS.pinned("b" * 40))
    assert other["job_id"] != base["job_id"]


def test_generation_cache_keys_bind_every_setting():
    items = plan_items(["a heron"], 1, seed=1)
    digest = items[0].request(SETTINGS).digest
    assert items[0].request(SETTINGS.pinned("b" * 40)).digest != digest
    assert items[0].request(GenerationSettings(**{**SETTINGS.as_json(), "steps": 3})).digest != digest


# ---- the run --------------------------------------------------------------
def test_a_clean_run_publishes_a_library_and_a_ledger(tmp_path):
    fake = FakeGenerator()
    status = run(tmp_path / "job", make_request(), fake)
    assert status["accepted"] == 6 and status["complete"] is True
    assert len(list((tmp_path / "job").glob("*.svg"))) == 6
    ledger = json.loads((tmp_path / "job/.job/ledger.json").read_text())
    assert {row["state"] for row in ledger["items"]} == {"accepted"}
    assert len(fake.calls) == 6


def test_resume_after_a_crash_keeps_completed_bytes_exactly(tmp_path):
    root = tmp_path / "job"
    request = make_request()
    crashing = FakeGenerator(crash_after=3)
    with pytest.raises(KeyboardInterrupt):
        run(root, request, crashing)
    done = sorted(root.glob("*.svg"))
    assert 0 < len(done) < 6
    before = {path.name: path.read_bytes() for path in done}

    resumed = FakeGenerator()
    status = run(root, request, resumed)
    assert status["accepted"] == 6
    after = {path.name: path.read_bytes() for path in root.glob("*.svg")}
    assert all(after[name] == data for name, data in before.items())
    # Completed work is not asked for a second time.
    assert len(resumed.calls) == 6 - len(before)


def test_tracing_resumes_without_asking_the_generator_again(tmp_path):
    """A conversion failure must not cost another GPU request."""
    root = tmp_path / "job"
    request = make_request(count=3, subjects=["a heron"])

    class Flaky(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.conversions = 0

        def convert(self, png, meta, item):
            self.conversions += 1
            if self.conversions <= 2:
                raise OSError("the tracer crashed")
            return super().convert(png, meta, item)

    flaky = Flaky()
    status = run(root, request, flaky, max_attempts=5)
    assert status["accepted"] == 3
    # Three images, five conversions: the retried rasters came off disk.
    assert len(flaky.calls) == 3 and flaky.conversions == 5


def test_a_corrupt_retained_raster_is_regenerated_not_accepted(tmp_path):
    root = tmp_path / "job"
    request = make_request(count=2, subjects=["a heron"])
    run(root, request, FakeGenerator())
    raw = next((root / ".job/raw").glob("*.png"))
    raw.write_bytes(b"\x89PNG\r\n\x1a\ntampered")

    fake = FakeGenerator()
    status = run(root, request, fake)
    assert status["accepted"] == 2
    assert len(fake.calls) == 1  # exactly the damaged one
    assert raw.read_bytes() != b"\x89PNG\r\n\x1a\ntampered"


def test_a_deleted_published_artifact_is_rebuilt(tmp_path):
    root = tmp_path / "job"
    request = make_request(count=2, subjects=["a heron"])
    run(root, request, FakeGenerator())
    victim = sorted(root.glob("*.svg"))[0]
    victim.unlink()
    fake = FakeGenerator()
    assert run(root, request, fake)["accepted"] == 2
    assert victim.is_file() and len(fake.calls) == 1


def test_a_refusal_is_terminal_and_counted_not_retried(tmp_path):
    fake = FakeGenerator(refuse={"a fern"})
    with Job.open(tmp_path / "job", make_request()) as job:
        job.run(generate=fake.generate, convert=fake.convert)
        report = job.report()
    assert report["accepted"] == 4 and report["refused"] == 2
    assert report["requested"] == 6
    assert not job.is_complete()


def test_transport_failures_are_retried_to_a_bound_then_recorded(tmp_path):
    fake = FakeGenerator(fail={"a fern"})
    with Job.open(tmp_path / "job", make_request()) as job:
        job.run(generate=fake.generate, convert=fake.convert, max_attempts=2)
        report = job.report()
    assert report["failed"] == 2 and report["accepted"] == 4
    assert sum(1 for call in fake.calls if call.subject == "a fern") == 4  # 2 items x 2 attempts


def test_identical_output_is_deduplicated_and_both_requests_are_kept(tmp_path):
    fake = FakeGenerator(repeat={"a heron": "same", "a fern": "same", "a koi": "same"})
    with Job.open(tmp_path / "job", make_request(count=3)) as job:
        job.run(generate=fake.generate, convert=fake.convert)
        report = job.report()
    assert report["accepted"] == 1 and report["duplicate"] == 2
    ledger = json.loads((tmp_path / "job/.job/ledger.json").read_text())
    assert sum(1 for row in ledger["items"] if row["duplicate_of"]) == 2
    assert len(list((tmp_path / "job").glob("*.svg"))) == 1


def test_a_replacement_budget_fills_a_refused_slot_without_adding_items(tmp_path):
    fake = FakeGenerator(refuse={"a fern"})

    class Once(FakeGenerator):
        def convert(self, png, meta, item):
            # Only the first candidate for the slot is refused; its replacement
            # is a different seed and passes.
            if item.subject == "a fern" and not item.replaces:
                raise ConversionRefusedError("nothing usable")
            return FakeGenerator.convert(self, png, meta, item)

    fake = Once()
    with Job.open(tmp_path / "job", make_request(count=3, budget=3)) as job:
        job.run(generate=fake.generate, convert=fake.convert)
        report = job.report()
    assert report["requested"] == 3 and report["accepted"] == 3
    assert report["refused"] == 1 and report["candidates"] == 4
    assert job.is_complete()


def test_cancellation_stops_asking_and_leaves_a_resumable_ledger(tmp_path):
    root = tmp_path / "job"
    request = make_request()
    fake = FakeGenerator()
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    with Job.open(root, request) as job:
        status = job.run(generate=fake.generate, convert=fake.convert, stop=stop)
    assert status["cancelled"] is True and status["accepted"] == 2
    assert not (root / "manifest.json").exists()
    assert run(root, request, FakeGenerator())["accepted"] == 6


def test_a_second_worker_gets_a_clear_busy_result(tmp_path):
    request = make_request()
    with Job.open(tmp_path / "job", request), pytest.raises(JobBusyError, match="another worker"):
        Job.open(tmp_path / "job", request).acquire()


def test_a_different_job_cannot_resume_in_an_occupied_directory(tmp_path):
    root = tmp_path / "job"
    run(root, make_request(count=2), FakeGenerator())
    with pytest.raises(Exception, match="resume the original"):
        Job.open(root, make_request(count=4))


def test_an_incomplete_job_publishes_a_selection_not_a_manifest(tmp_path):
    fake = FakeGenerator(refuse={"a fern"})
    root = tmp_path / "job"
    with Job.open(root, make_request()) as job:
        job.run(generate=fake.generate, convert=fake.convert)
        selection = job.finalize_selection(reason="operator accepted the usable subset")
    assert not (root / "manifest.json").exists()
    assert selection["complete"] is False
    assert (selection["requested"], selection["accepted"], selection["refused"]) == (6, 4, 2)


def test_a_missing_backend_makes_zero_requests(tmp_path):
    """Bookkeeping only: the run never reaches a generator it cannot name."""
    def refuse(_request):
        raise AssertionError("the generator must not be contacted")

    with Job.open(tmp_path / "job", make_request(count=2)) as job:
        status = job.run(generate=refuse, convert=lambda *_: {}, max_attempts=1)
    assert status["accepted"] == 0 and status["counts"]["failed"] == 2


def test_the_frozen_request_is_readable_for_resume(tmp_path):
    root = tmp_path / "job"
    request = make_request(count=2)
    run(root, request, FakeGenerator())
    assert load_job(root)["job_id"] == request["job_id"]


def test_a_ledger_from_another_job_is_refused(tmp_path):
    root = tmp_path / "job"
    request = make_request(count=2)
    run(root, request, FakeGenerator())
    ledger = json.loads((root / ".job/ledger.json").read_text())
    ledger["job_id"] = "0" * 64
    (root / ".job/ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(Exception, match="different job"):
        Job.open(root, request).acquire()


def test_artifacts_never_escape_the_job_directory(tmp_path):
    def escape(png, meta, item):
        return {"artifacts": {"../escaped.svg": "<svg/>"}, "record": {}}

    with Job.open(tmp_path / "job", make_request(count=1, subjects=["a heron"])) as job:
        job.run(generate=FakeGenerator().generate, convert=escape, max_attempts=1)
    assert not (tmp_path / "escaped.svg").exists()
    assert os.listdir(tmp_path / "job") == [".job"]


def test_a_changed_conversion_retraces_without_asking_the_generator_again(tmp_path):
    """The generation cache is keyed by the request; the conversion is not.

    Rerunning a job at a different artwork size used to keep the old artwork:
    every item was already `accepted`, so nothing reconverted. The raster is
    still valid — only the tracing has to happen again, and it costs no GPU.
    """
    root = tmp_path / "job"
    request = make_request(count=3, subjects=["a heron"])
    first = FakeGenerator()
    with Job.open(root, request) as job:
        job.run(generate=first.generate, convert=first.convert, conversion_key="size=50")
    assert len(first.calls) == 3
    before = {path.name: path.read_bytes() for path in root.glob("*.svg")}

    class Wider(FakeGenerator):
        def convert(self, png, meta, item):
            result = FakeGenerator.convert(self, png, meta, item)
            result["artifacts"] = {name: (f"<svg size='20' data='{item.key}'/>" if name.endswith(".svg") else data)
                                   for name, data in result["artifacts"].items()}
            return result

    second = Wider()
    with Job.open(root, request) as job:
        status = job.run(generate=second.generate, convert=second.convert, conversion_key="size=20")
    assert status["accepted"] == 3
    assert second.calls == []  # not one request reached the generator
    after = {path.name: path.read_bytes() for path in root.glob("*.svg")}
    assert set(after) == set(before)
    assert all(after[name] != data for name, data in before.items())

    # And running again under the same key changes nothing at all.
    third = Wider()
    with Job.open(root, request) as job:
        job.run(generate=third.generate, convert=third.convert, conversion_key="size=20")
    assert third.calls == []
    assert {p.name: p.read_bytes() for p in root.glob("*.svg")} == after


def test_no_conversion_key_keeps_the_previous_behaviour(tmp_path):
    root = tmp_path / "job"
    request = make_request(count=2, subjects=["a heron"])
    run(root, request, FakeGenerator())
    fake = FakeGenerator()
    assert run(root, request, fake)["accepted"] == 2
    assert fake.calls == []
