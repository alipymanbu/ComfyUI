"""The startup prune keeps a row whose folder is still registered under another
spelling (a symlink or other alias, or letter case), rewriting its path to today's
spelling instead of retiring it. Boots are the prune followed by the seeder's real
fast phase, on an in-memory catalog."""

import logging
import os
import posixpath
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

import folder_paths
from app.assets import scanner_rehome, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.scanner import mark_contents_missing_outside_prefixes
from app.assets.scanner_rehome import (
    CaseRespeller,
    PrunePlan,
    PruneRow,
    apply_prune_plan,
    plan_prune,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="builds symlinks and simulates case folding on POSIX")

ALL_ROOTS = ("models", "input", "output")
OUTPUT_FILES = ("a.png", "b.png", os.path.join("sub", "c.png"), os.path.join("sub", "deep", "d.png"))
MODEL_FILES = ("m1.safetensors", os.path.join("sd", "m2.safetensors"))


@pytest.fixture(autouse=True)
def isolated_state(db_engine):
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    _WATCH_LIST.clear()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield
    _WATCH_LIST.clear()


class Folders:
    """The folder configuration one boot sees; ``use`` switches it between boots."""

    def __init__(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        self.base = temp_dir
        self._monkeypatch = monkeypatch
        for name in ("input", "temp"):
            (temp_dir / name).mkdir()
        monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(temp_dir / "input"))
        monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_dir / "temp"))
        self.use(output=None, models=None)

    def use(self, *, output: Path | None, models: Path | None, input: Path | None = None) -> None:
        input_dir = str(input if input is not None else self.base / "input")
        self._monkeypatch.setattr(folder_paths, "get_input_directory", lambda: input_dir)
        output_dir = str(output if output is not None else self.base / "no-output")
        self._monkeypatch.setattr(folder_paths, "get_output_directory", lambda: output_dir)
        registered = {} if models is None else {"checkpoints": ([str(models)], {".safetensors"})}
        self._monkeypatch.setattr(folder_paths, "folder_names_and_paths", registered)
        self._monkeypatch.setattr(folder_paths, "filename_list_cache", {})


@pytest.fixture
def folders(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Folders:
    return Folders(temp_dir, monkeypatch)


def _populate(root: Path, names: tuple[str, ...]) -> None:
    for i, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (i + 1))


def _alias(target: Path, link: Path) -> Path:
    link.symlink_to(target, target_is_directory=True)
    return link


def _case_variant(target: Path, variant: Path) -> Path:
    """``variant`` (``target`` in other letter case) reaching ``target``: as-is on a
    case-insensitive filesystem (macOS by default), through a symlink elsewhere."""
    if variant.exists():
        return variant
    return _alias(target, variant)


def _boot(caplog: pytest.LogCaptureFixture | None = None) -> int:
    """One startup: the prune, then the fast phase. Returns the rows it created."""
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._roots = ALL_ROOTS
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._prune_first = True
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    created: list[int] = []
    run_fast_phase = seeder._run_fast_phase

    def record_created(roots):
        result = run_fast_phase(roots)
        created.append(result[0])
        return result

    seeder._run_fast_phase = record_created
    if caplog is not None:
        caplog.clear()
        with caplog.at_level(logging.INFO):
            seeder._run_scan()
    else:
        seeder._run_scan()
    assert seeder._errors == []
    return created[0]


def _records(session) -> dict[str, tuple[str, str]]:
    """record id -> (content path, name), live rows only."""
    session.expire_all()
    rows = session.execute(
        sa.select(Asset.id, AssetContent.path, Asset.name)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(AssetContent.is_missing.is_(False))
    )
    return {record_id: (path, name) for record_id, path, name in rows}


def _missing_count(session) -> int:
    session.expire_all()
    return session.scalar(
        sa.select(sa.func.count()).select_from(AssetContent).where(AssetContent.is_missing.is_(True))
    )


def _rename_all(session) -> None:
    for record in session.scalars(sa.select(Asset)):
        record.name = f"kept-{record.id}"
    session.commit()


def _marked_missing_event(caplog: pytest.LogCaptureFixture) -> dict[str, str]:
    lines = [r.getMessage() for r in caplog.records if "seeder.marked_missing" in r.getMessage()]
    assert len(lines) == 1, lines
    return dict(pair.split("=", 1) for pair in lines[0].split()[2:])


def _respelled(records: dict[str, tuple[str, str]], old: Path, new: Path) -> dict[str, tuple[str, str]]:
    return {rid: (path.replace(str(old), str(new), 1), name) for rid, (path, name) in records.items()}


def test_output_seeded_through_a_symlink_keeps_its_records_when_booted_by_the_real_path(
    folders, temp_dir, session, caplog
):
    real = temp_dir / "real" / "output"
    _populate(real, OUTPUT_FILES)
    alias = _alias(temp_dir / "real", temp_dir / "alias") / "output"
    folders.use(output=alias, models=None)
    assert _boot() == len(OUTPUT_FILES)
    _rename_all(session)
    before = _records(session)

    folders.use(output=real, models=None)
    created = _boot(caplog)

    assert created == 0
    assert _records(session) == _respelled(before, alias, real)
    assert _missing_count(session) == 0
    assert _marked_missing_event(caplog) == {
        "count": "0",
        "rehomed_count": str(len(OUTPUT_FILES)),
        "still_present_count": "0",
        "conflict_retired_count": "0",
        "stage": "pruning",
    }

    assert _boot(caplog) == 0
    assert _marked_missing_event(caplog)["rehomed_count"] == "0"
    assert _records(session) == _respelled(before, alias, real)


def test_a_model_folder_registered_through_an_alias_keeps_its_records(folders, temp_dir, session):
    real = temp_dir / "disk" / "checkpoints"
    _populate(real, MODEL_FILES)
    alias = _alias(temp_dir / "disk", temp_dir / "mnt") / "checkpoints"
    folders.use(output=None, models=alias)
    assert _boot() == len(MODEL_FILES)
    _rename_all(session)
    before = _records(session)

    folders.use(output=None, models=real)

    assert _boot() == 0
    assert _records(session) == _respelled(before, alias, real)
    assert _missing_count(session) == 0


def test_a_different_folder_with_the_same_layout_is_not_rehomed(folders, temp_dir, session):
    old = temp_dir / "old" / "output"
    new = temp_dir / "new" / "output"
    _populate(old, OUTPUT_FILES)
    _populate(new, OUTPUT_FILES)
    folders.use(output=old, models=None)
    _boot()
    old_ids = set(_records(session))

    folders.use(output=new, models=None)

    assert _boot() == len(OUTPUT_FILES)
    assert not old_ids & set(_records(session))
    assert _missing_count(session) == len(OUTPUT_FILES)


@pytest.fixture
def folds_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare paths the way Windows does. The filesystem stays case-sensitive, so the
    tests reach one folder under two spellings through a symlink."""
    monkeypatch.setattr(posixpath, "normcase", lambda path: os.fspath(path).lower())


def test_a_case_only_respelling_neither_duplicates_nor_retires(folders, folds_case, temp_dir, session, caplog):
    real = temp_dir / "data" / "output"
    _populate(real, OUTPUT_FILES)
    upper = _case_variant(temp_dir / "data", temp_dir / "DATA") / "output"
    folders.use(output=upper, models=None)
    _boot()
    _rename_all(session)
    before = _records(session)

    folders.use(output=real, models=None)

    assert _boot(caplog) == 0
    assert _records(session) == _respelled(before, upper, real)
    assert _missing_count(session) == 0
    assert _marked_missing_event(caplog)["rehomed_count"] == str(len(OUTPUT_FILES))


def test_existing_case_duplicates_heal_keeping_the_older_row(folders, folds_case, temp_dir, session, caplog):
    real = temp_dir / "data" / "output"
    _populate(real, OUTPUT_FILES)
    upper = _case_variant(temp_dir / "data", temp_dir / "DATA") / "output"
    folders.use(output=upper, models=None)
    _boot()
    _rename_all(session)
    before = _records(session)
    # What a case-only relaunch left behind before this fix: a newer live duplicate of every file.
    later = datetime.now() + timedelta(days=1)
    for name in OUTPUT_FILES:
        content = create_content(session, path=str(real / name), size_bytes=1, mtime_ns=1)
        content.created_at = later
        create_record(session, content_id=content.id, name="duplicate")
    session.commit()

    folders.use(output=real, models=None)

    assert _boot(caplog) == 0
    assert _records(session) == _respelled(before, upper, real)
    assert _missing_count(session) == len(OUTPUT_FILES)
    event = _marked_missing_event(caplog)
    assert event["conflict_retired_count"] == str(len(OUTPUT_FILES))
    assert event["rehomed_count"] == str(len(OUTPUT_FILES))


# --- the plan, row by row ----------------------------------------------------------------

T0 = datetime(2026, 1, 1)


def _row(session, path: Path, age_days: int = 0) -> PruneRow:
    """A row catalogued as an output file directly under the output folder."""
    content = create_content(session, path=str(path), size_bytes=1, mtime_ns=1)
    content.created_at = T0 + timedelta(days=age_days)
    create_record(session, content_id=content.id, name=path.name, loader_path=path.name, tags=["output"])
    session.flush()
    return PruneRow(content.id, content.path, content.created_at)


def _live(session, content_id: str) -> str | None:
    session.expire_all()
    content = session.get(AssetContent, content_id)
    return None if content.is_missing else content.path


def test_a_gone_file_is_retired_and_not_counted_as_present(session, temp_dir):
    (temp_dir / "old").mkdir()
    row = _row(session, temp_dir / "old" / "gone.png")

    plan = plan_prune([], [row], [str(temp_dir / "new")])

    assert plan == PrunePlan([], [row.content_id], 0)


def test_a_file_outside_every_owned_folder_is_retired_and_counted_as_present(session, temp_dir):
    _populate(temp_dir / "old", ("f.png",))
    (temp_dir / "new").mkdir()
    row = _row(session, temp_dir / "old" / "f.png")

    result = apply_prune_plan(session, plan_prune([], [row], [str(temp_dir / "new")]))

    assert (result.marked, result.rehomed, result.still_present) == (1, 0, 1)


def test_an_unreadable_parent_skips_the_per_file_stat(session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png", "g.png"))
    alias = _alias(real, temp_dir / "share")
    rows = [_row(session, alias / "f.png"), _row(session, alias / "g.png")]
    real_stat = os.stat
    stat_calls: list[str] = []

    def offline_share(path, *args, **kwargs):
        stat_calls.append(os.fspath(path))
        if os.fspath(path) == str(alias):
            raise OSError(112, "host is down")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", offline_share)

    plan = plan_prune([], rows, [str(real)])

    assert plan == PrunePlan([], [row.content_id for row in rows], 0)
    assert stat_calls == [str(alias)]


def test_a_filesystem_without_inode_numbers_is_never_rehomed(session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    alias = _alias(real, temp_dir / "share")
    row = _row(session, alias / "f.png")
    real_stat = os.stat

    def no_inodes(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        fields = list(result)
        fields[1] = 0  # st_ino
        return os.stat_result(fields)

    monkeypatch.setattr(os, "stat", no_inodes)

    assert plan_prune([], [row], [str(real)]) == PrunePlan([], [row.content_id], 1)


def test_an_older_live_row_at_the_new_path_keeps_it(folders, session, temp_dir):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    alias = _alias(real, temp_dir / "alias")
    occupant = _row(session, real / "f.png", age_days=0)
    mover = _row(session, alias / "f.png", age_days=1)

    result = apply_prune_plan(session, plan_prune([], [mover], [str(real)]))

    assert (result.marked, result.rehomed, result.conflict_retired) == (1, 0, 1)
    assert _live(session, occupant.content_id) == str(real / "f.png")
    assert _live(session, mover.content_id) is None


def test_of_two_aliases_for_one_file_the_older_row_moves(folders, session, temp_dir):
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    folders.use(output=real, models=None)
    newer = _row(session, _alias(real, temp_dir / "a1") / "f.png", age_days=2)
    older = _row(session, _alias(real, temp_dir / "a2") / "f.png", age_days=1)

    result = apply_prune_plan(session, plan_prune([], [newer, older], [str(real)]))

    assert (result.marked, result.rehomed, result.conflict_retired) == (1, 1, 1)
    assert _live(session, older.content_id) == str(real / "f.png")
    assert _live(session, newer.content_id) is None


def test_case_respelling_uses_the_deepest_matching_prefix(folds_case):
    respell = CaseRespeller(["/Data", "/Data/Output"])

    assert respell("/data/output/sub/F.png") == ("/Data/Output/sub/F.png", "/Data/Output")
    assert respell("/data/other/F.png") == ("/Data/other/F.png", "/Data")
    assert respell("/elsewhere/F.png") is None


def test_a_row_spelled_for_a_shallow_prefix_is_respelled_for_a_deeper_one(folds_case, session, temp_dir):
    shallow = temp_dir / "models"
    (shallow / "output").mkdir(parents=True)
    _case_variant(shallow / "output", shallow / "Output")
    row = _row(session, shallow / "output" / "f.png")

    result = mark_contents_missing_outside_prefixes(session, [str(shallow), str(shallow / "Output")])

    assert (result.marked, result.rehomed) == (0, 1)
    assert _live(session, row.content_id) == str(shallow / "Output" / "f.png")


def test_without_case_folding_owned_rows_are_left_alone(session, temp_dir):
    shallow = temp_dir / "models"
    row = _row(session, shallow / "output" / "f.png")

    result = mark_contents_missing_outside_prefixes(session, [str(shallow), str(shallow / "Output")])

    assert (result.marked, result.rehomed) == (0, 0)
    assert _live(session, row.content_id) == str(shallow / "output" / "f.png")


def _folds_case_on_disk(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.mkdir()
    try:
        return (directory / "CASE-PROBE").exists()
    finally:
        probe.rmdir()


def test_a_case_sensitive_sibling_folder_is_not_mistaken_for_the_same_one(
    folders, folds_case, temp_dir, session, caplog
):
    """Windows can mark a directory case-sensitive (WSL does), so output and Output
    are different folders even where paths are compared case-insensitively."""
    if _folds_case_on_disk(temp_dir):
        pytest.skip("this filesystem cannot hold two names differing only in case")
    lower = temp_dir / "cs" / "output"
    upper = temp_dir / "cs" / "Output"
    _populate(lower, OUTPUT_FILES)
    _populate(upper, OUTPUT_FILES)  # same names and sizes, different files
    folders.use(output=lower, models=None)
    _boot()
    _rename_all(session)
    old_ids = set(_records(session))

    folders.use(output=upper, models=None)

    assert _boot(caplog) == len(OUTPUT_FILES)
    assert not old_ids & set(_records(session))
    assert _missing_count(session) == len(OUTPUT_FILES)
    event = _marked_missing_event(caplog)
    assert event["rehomed_count"] == "0"
    assert event["still_present_count"] == str(len(OUTPUT_FILES))


def test_a_case_respelling_whose_folders_cannot_be_compared_is_left_alone(folds_case, session, temp_dir):
    stored = temp_dir / "offline" / "output"
    row = _row(session, stored / "f.png")

    result = mark_contents_missing_outside_prefixes(session, [str(temp_dir / "Offline" / "output")])

    assert result == (0, 0, 0, 0)
    assert _live(session, row.content_id) == str(stored / "f.png")


def test_an_unreachable_folder_is_probed_once_then_never_again(session, temp_dir, monkeypatch):
    """Retired rows are not live, and the prune selects only live rows, so a folder on
    an unreachable host costs its timeout on one startup only."""
    gone = temp_dir / "unreachable-host" / "output"
    rows = [_row(session, gone / "f.png"), _row(session, gone / "g.png")]
    owned = [str(temp_dir / "output")]
    real_stat = os.stat
    probed: list[str] = []

    def unreachable(path, *args, **kwargs):
        if os.fspath(path).startswith(str(temp_dir / "unreachable-host")):
            probed.append(os.fspath(path))
            raise OSError(53, "network path not found")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", unreachable)

    first = mark_contents_missing_outside_prefixes(session, owned)
    session.commit()
    first_probes = len(probed)
    second = mark_contents_missing_outside_prefixes(session, owned)

    assert first.marked == len(rows)
    assert first_probes == 1  # the shared parent, once
    assert second == (0, 0, 0, 0)
    assert len(probed) == first_probes


# --- QA findings: hung mounts, different files, the row's own root, racing writers ----

def test_a_hung_mount_costs_a_bounded_wait_not_the_seeder(folders, session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("ok.png",))
    folders.use(output=real, models=None)
    kept = _row(session, _alias(real, temp_dir / "alias") / "ok.png")
    hung = [_row(session, temp_dir / "dead" / d / "f.png") for d in ("a", "b")]
    monkeypatch.setattr(scanner_rehome, "STALL_SECONDS", 0.2)
    release = threading.Event()
    real_stat = os.stat

    def hard_mount(path, *args, **kwargs):
        if os.fspath(path).startswith(str(temp_dir / "dead")):
            release.wait(30)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", hard_mount)
    try:
        started = time.monotonic()
        result = mark_contents_missing_outside_prefixes(session, [str(real)])
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 5
    assert (result.marked, result.rehomed, result.still_present) == (2, 1, 0)
    assert [_live(session, row.content_id) for row in hung] == [None, None]
    assert _live(session, kept.content_id) == str(real / "ok.png")


def test_a_different_file_behind_the_same_folder_is_not_rehomed(folders, session, temp_dir, monkeypatch):
    """A relative ../ symlink inside a bind-mounted folder names a different file under
    each spelling, though the folders compare equal. Simulated: the new path stats as
    another file."""
    real = temp_dir / "real"
    _populate(real, ("f.png",))
    _populate(temp_dir / "stranger", ("f.png",))
    folders.use(output=real, models=None)
    row = _row(session, _alias(real, temp_dir / "alias") / "f.png")
    real_stat = os.stat

    def other_file(path, *args, **kwargs):
        if os.fspath(path) == str(real / "f.png"):
            return real_stat(temp_dir / "stranger" / "f.png")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", other_file)

    result = mark_contents_missing_outside_prefixes(session, [str(real)])

    assert (result.marked, result.rehomed, result.still_present) == (1, 0, 1)
    assert _live(session, row.content_id) is None


def test_a_model_folder_inside_the_output_folder_keeps_output_rows_as_output(folders, temp_dir, session):
    out = temp_dir / "real" / "out"
    _populate(out, ("checkpoints/m.safetensors", "a.png"))
    folders.use(output=_alias(temp_dir / "real", temp_dir / "A") / "out", models=None)
    _boot()
    _rename_all(session)
    before = _records(session)
    checkpoints = _alias(out / "checkpoints", temp_dir / "L")

    folders.use(output=out, models=checkpoints)
    _boot()

    after = _records(session)
    for record_id, (path, name) in before.items():
        assert after[record_id] == (path.replace(str(temp_dir / "A" / "out"), str(out)), name)
    session.expire_all()
    records = session.scalars(sa.select(Asset).where(Asset.id.in_(before))).all()
    assert {record.loader_path for record in records} == {"checkpoints/m.safetensors", "a.png"}


def test_input_and_output_on_one_folder_keep_each_row_in_its_own_root(folders, temp_dir, session):
    shared = temp_dir / "shared"
    _populate(shared, ("f.png",))
    folders.use(output=_alias(shared, temp_dir / "X1"), models=None)
    _boot()
    _rename_all(session)
    (record_id,) = _records(session)

    folders.use(output=shared, models=None, input=_alias(shared, temp_dir / "Y"))
    _boot()

    assert _records(session)[record_id][0] == str(shared / "f.png")


def test_a_row_whose_folder_now_has_only_another_role_is_retired(folders, temp_dir, session, caplog):
    out = temp_dir / "real" / "out"
    _populate(out, ("m.safetensors",))
    folders.use(output=_alias(temp_dir / "real", temp_dir / "A") / "out", models=None)
    _boot()
    old_ids = set(_records(session))

    folders.use(output=None, models=_alias(out, temp_dir / "L"))
    _boot(caplog)

    assert not old_ids & set(_records(session))
    event = _marked_missing_event(caplog)
    assert (event["rehomed_count"], event["still_present_count"]) == ("0", "1")


def test_a_target_taken_by_a_racing_writer_skips_that_row_only(folders, session, temp_dir, monkeypatch):
    real = temp_dir / "real"
    _populate(real, ("f.png", "g.png"))
    folders.use(output=real, models=None)
    alias = _alias(real, temp_dir / "alias")
    raced = _row(session, alias / "f.png")
    moved = _row(session, alias / "g.png")
    racer = _row(session, real / "f.png", age_days=5)
    # The racer's insert lands between the occupant read and the rewrite.
    monkeypatch.setattr(scanner_rehome, "_live_occupants", lambda _session, _paths: {})

    result = mark_contents_missing_outside_prefixes(session, [str(real)])

    assert (result.marked, result.rehomed) == (0, 1)
    assert _live(session, raced.content_id) == str(alias / "f.png")
    assert _live(session, moved.content_id) == str(real / "g.png")
    assert _live(session, racer.content_id) == str(real / "f.png")
