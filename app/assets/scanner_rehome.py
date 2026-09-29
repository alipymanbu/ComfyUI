"""Keeps a catalogued row through the startup prune when its folder is still owned
but spelled differently today: an 8.3 short name, a junction, a subst drive, a
``\\\\?\\`` prefix, a symlink or bind mount, or letter case. The prune rewrites such
a row's folder part to today's spelling instead of retiring it, so the row keeps
its id, name, tags and job links, and the scan that follows finds it at the path
it lists.

Every live row then carries today's exact spelling, which is what lets the
per-root sync, the scan's existing-path dedupe and the live-path unique index
stay exact-string comparisons.

Sameness is decided by stat (device and inode), never by text alone:

- A case-only respelling needs the stored and current folder spellings to be one
  directory. A case-sensitive directory on Windows (or one made by WSL) can hold
  ``output`` and ``Output`` side by side.
- Any other respelling needs the file at the new path to be the same file as the
  one at the old path. A ``../`` symlink inside a bind-mounted folder resolves
  differently under each spelling.
- The new path must lie in a folder of the row's own role: the tags and loader
  path it would be given there must be the ones it already has. One directory can
  be registered as, say, both a model folder and part of the output folder.

A row whose sameness cannot be decided is treated as the prune treated it before:
retired when it fails the text match, left alone when it passes it. Filesystem
reads run on a worker thread under a watchdog, so a hung mount (a hard NFS mount
whose server is gone, a stopped FUSE daemon) costs a bounded wait, not the seeder.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import NamedTuple

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import mark_content_missing
from app.assets.services.gil import yield_gil
from app.assets.services.path_utils import compute_loader_path, get_path_derived_tags_from_path

_BATCH = 500

# A worker that completes no filesystem call for this long is taken to be stuck on a
# hung mount. The rows it was checking are treated as undecided and a fresh worker
# carries on. After this many stalls the rest are undecided without being checked,
# so a dead mount costs at most STALL_SECONDS * MAX_STALLS.
STALL_SECONDS = 10.0
MAX_STALLS = 3

_Identity = tuple[int, int]


class PruneResult(NamedTuple):
    marked: int  # rows retired, conflict losers included
    rehomed: int  # rows rewritten to today's spelling of their folder
    still_present: int  # retired rows whose file still stats
    conflict_retired: int  # retired because an older row holds the same path


class PruneRow(NamedTuple):
    content_id: str
    path: str
    created_at: datetime


class RespelledRow(NamedTuple):
    row: PruneRow
    new_path: str
    prefix: str  # the owning prefix, as spelled today


class Move(NamedTuple):
    row: PruneRow
    candidates: tuple[str, ...]  # the same file under owned folders, deepest first
    check_role: bool  # take the first candidate in the row's own role


class PrunePlan(NamedTuple):
    moves: list[Move]
    retire: list[str]
    still_present: int


class CaseRespeller:
    """Maps a path to its folder part in the spelling of the deepest prefix it lies
    under by the platform's case rules. Pure string; plan_prune confirms the two
    spellings are one folder before acting on it. The prefixes are normalised once,
    since the prune calls this for every live row."""

    def __init__(self, prefixes: Sequence[str]):
        stems: list[tuple[str, str, str]] = []
        for prefix in {os.path.abspath(p) for p in prefixes}:
            folded = os.path.normcase(prefix)
            if len(folded) != len(prefix):  # case mapping changed the length: no safe splice
                continue
            stem = folded if folded.endswith(os.sep) else folded + os.sep
            stems.append((prefix, folded, stem))
        stems.sort(key=lambda entry: len(entry[0]), reverse=True)
        self._stems = stems
        self.folds_case = os.path.normcase("A") != "A"

    def __call__(self, path: str) -> tuple[str, str] | None:
        """(``path`` respelled, the prefix it lies under), or None when no prefix
        contains it. Deepest prefix first, so a row exactly under a shallow prefix is
        still respelled for a deeper one."""
        candidate = os.path.normcase(path)
        for prefix, folded, stem in self._stems:
            if candidate == folded or candidate.startswith(stem):
                return prefix + path[len(prefix):], prefix
        return None


# --- filesystem reads --------------------------------------------------------------------

_progress = threading.local()


def _stat_or_none(path: str) -> os.stat_result | None:
    try:
        return os.stat(path)
    except (OSError, ValueError):
        return None
    finally:
        tick = getattr(_progress, "tick", None)
        if tick is not None:
            tick()


def _identity(stat_result: os.stat_result | None, *, directory: bool) -> _Identity | None:
    # Inode 0 means the filesystem (some SMB and FUSE mounts) reports no file ids,
    # so everything on it would compare equal.
    if stat_result is None or stat_result.st_ino == 0:
        return None
    if directory and not stat.S_ISDIR(stat_result.st_mode):
        return None
    return (stat_result.st_dev, stat_result.st_ino)


class _FolderResolver:
    """Stats each directory at most once, so the cost is per distinct folder, not per row."""

    def __init__(self, prefixes: Sequence[str]):
        self._prefixes = list(dict.fromkeys(os.path.abspath(p) for p in prefixes))
        self._stats: dict[str, os.stat_result | None] = {}
        self._owned: dict[_Identity, list[str]] | None = None
        self._owners: dict[str, tuple[tuple[str, str], ...]] = {}

    def dir_stat(self, directory: str) -> os.stat_result | None:
        if directory not in self._stats:
            self._stats[directory] = _stat_or_none(directory)
        return self._stats[directory]

    def _owned_identities(self) -> dict[_Identity, list[str]]:
        if self._owned is None:
            owned: dict[_Identity, list[str]] = {}
            for prefix in self._prefixes:
                identity = _identity(self.dir_stat(prefix), directory=True)
                if identity is not None:
                    owned.setdefault(identity, []).append(prefix)
            self._owned = owned
        return self._owned

    def owners(self, directory: str) -> tuple[tuple[str, str], ...]:
        """Every (ancestor, prefix) where an ancestor of ``directory`` (itself
        included) is the same folder as an owned prefix, deepest ancestor first."""
        owned = self._owned_identities()
        chain: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        current = directory
        while current not in self._owners:
            identity = _identity(self.dir_stat(current), directory=True)
            matches = tuple((current, prefix) for prefix in owned.get(identity, ())) if identity else ()
            parent = os.path.dirname(current)
            if parent == current:  # the filesystem root
                self._owners[current] = matches
                break
            chain.append((current, matches))
            current = parent
        found = self._owners[current]
        for walked, matches in reversed(chain):
            found = matches + found
            self._owners[walked] = found
        return self._owners[directory]

    def same_folder(self, first: str, second: str) -> bool | None:
        """Whether two directories are one, by device and inode; None when a stat
        fails or the filesystem reports no inode numbers."""
        first_id = _identity(self.dir_stat(first), directory=True)
        second_id = _identity(self.dir_stat(second), directory=True)
        if first_id is None or second_id is None:
            return None
        return first_id == second_id


class _Unknown:
    """A job whose worker stalled, or that never ran because too many had."""


_UNKNOWN = _Unknown()


class _Worker:
    def __init__(self, jobs: Sequence[Callable[[], object]], start: int, results: list, lock: threading.Lock):
        self._jobs = jobs
        self._results = results
        self._lock = lock
        self.index = start
        self.last_progress = time.monotonic()
        self.abandoned = False
        self.error: BaseException | None = None
        self.done = threading.Event()

    def _tick(self) -> None:
        self.last_progress = time.monotonic()

    def run(self) -> None:
        _progress.tick = self._tick
        try:
            for index in range(self.index, len(self._jobs)):
                with self._lock:
                    if self.abandoned:
                        return
                    self.index = index
                    self.last_progress = time.monotonic()
                result = self._jobs[index]()
                with self._lock:
                    if self.abandoned:
                        return
                    self._results[index] = result
        except BaseException as error:  # handed to the waiting thread, which re-raises it
            self.error = error
        finally:
            self.done.set()


class _Watchdog:
    """Runs jobs on a worker thread, replacing a worker that stops making progress.

    A stuck thread cannot be interrupted, so it is abandoned (a daemon thread,
    parked in the kernel) and its results are ignored if it ever returns.
    """

    def __init__(self) -> None:
        self.stalls = 0

    def run(self, jobs: Sequence[Callable[[], object]]) -> list:
        results: list = [_UNKNOWN] * len(jobs)
        lock = threading.Lock()
        start = 0
        while start < len(jobs) and self.stalls < MAX_STALLS:
            worker = _Worker(jobs, start, results, lock)
            threading.Thread(target=worker.run, name="asset-prune-io", daemon=True).start()
            while not worker.done.wait(timeout=STALL_SECONDS / 4):
                if time.monotonic() - worker.last_progress > STALL_SECONDS:
                    with lock:
                        worker.abandoned = True
                        start = worker.index + 1
                    self.stalls += 1
                    logging.warning(
                        "Asset prune: a filesystem check made no progress for %.0fs "
                        "(a hung mount?); treating those references as unknown",
                        STALL_SECONDS,
                    )
                    break
            else:
                if worker.error is not None:
                    raise worker.error
                start = len(jobs)
        if start < len(jobs):
            logging.warning(
                "Asset prune: skipped filesystem checks for %d folders after %d stalls",
                len(jobs) - start,
                self.stalls,
            )
        return results


# --- planning ----------------------------------------------------------------------------

class _RowOutcome(NamedTuple):
    row: PruneRow
    candidates: tuple[str, ...]  # empty: retire
    present: bool  # the file still stats at the old path


def _check_unowned(resolver: _FolderResolver, rows: Sequence[PruneRow]) -> list[_RowOutcome]:
    """One parent directory's rows. A parent that cannot be stat'ed (an unplugged
    drive, an offline share) skips the per-file stats, so an offline folder costs one
    failed stat, not one per file."""
    parent = os.path.dirname(rows[0].path)
    if resolver.dir_stat(parent) is None:
        return [_RowOutcome(row, (), False) for row in rows]
    outcomes = []
    for row in rows:
        yield_gil()
        old_stat = _stat_or_none(row.path)
        old_identity = _identity(old_stat, directory=False)
        candidates: list[str] = []
        # Without an identity (the file is gone, or the filesystem has no inode
        # numbers) nothing can be proven the same file, so there are no candidates.
        if old_identity is not None:
            for ancestor, prefix in resolver.owners(parent):
                new_path = os.path.join(prefix, row.path[len(ancestor):].lstrip(os.sep))
                if new_path in candidates:
                    continue
                if _identity(_stat_or_none(new_path), directory=False) == old_identity:
                    candidates.append(new_path)
        outcomes.append(_RowOutcome(row, tuple(candidates), old_stat is not None))
    return outcomes


def plan_prune(
    respelled: Iterable[RespelledRow],
    unowned: Iterable[PruneRow],
    prefixes: Sequence[str],
) -> PrunePlan:
    """Decide each row's fate with filesystem reads only; nothing is written.

    ``respelled`` rows are owned by case-folded text but spelled differently from
    their deepest prefix, already paired with CaseRespeller's path. ``unowned`` rows
    fail the text match altogether.
    """
    moves: list[Move] = []
    retire: list[str] = []
    still_present = 0
    resolver = _FolderResolver(prefixes)
    watchdog = _Watchdog()
    unowned = list(unowned)

    by_spelling: dict[tuple[str, str], list[RespelledRow]] = {}
    for respelled_row in respelled:
        stored = respelled_row.row.path[:len(respelled_row.prefix)]
        by_spelling.setdefault((stored, respelled_row.prefix), []).append(respelled_row)
    spellings = list(by_spelling)
    verdicts = watchdog.run([lambda pair=pair: resolver.same_folder(*pair) for pair in spellings])
    for pair, same in zip(spellings, verdicts):
        for respelled_row in by_spelling[pair]:
            if same is True:
                moves.append(Move(respelled_row.row, (respelled_row.new_path,), check_role=False))
            elif same is False:
                # A case-sensitive directory: another folder that differs only in case.
                # Treat the row as unowned; a real alias can still claim it.
                unowned.append(respelled_row.row)
            # Undecided (a stat failed or stalled, or no inode numbers): leave the row as
            # it is, as the prune did before re-spelling existed.

    by_parent: dict[str, list[PruneRow]] = {}
    for row in unowned:
        by_parent.setdefault(os.path.dirname(row.path), []).append(row)
    groups = list(by_parent.values())
    checked = watchdog.run([lambda rows=rows: _check_unowned(resolver, rows) for rows in groups])
    for rows, outcomes in zip(groups, checked):
        if outcomes is _UNKNOWN:
            retire.extend(row.content_id for row in rows)
            continue
        for outcome in outcomes:
            if outcome.candidates:
                moves.append(Move(outcome.row, outcome.candidates, check_role=True))
            else:
                retire.append(outcome.row.content_id)
                still_present += outcome.present
    return PrunePlan(moves, retire, still_present)


# --- applying ----------------------------------------------------------------------------

_RecordRole = tuple[str | None, frozenset[str]]  # (loader_path, tags)


def _batches(items: Sequence, size: int = _BATCH) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _record_roles(session: Session, content_ids: Sequence[str]) -> dict[str, list[_RecordRole]]:
    roles: dict[str, list[_RecordRole]] = {}
    for chunk in _batches(list(content_ids)):
        records = session.execute(
            sa.select(Asset.id, Asset.content_id, Asset.loader_path).where(Asset.content_id.in_(chunk))
        ).all()
        tags: dict[str, set[str]] = {}
        for record_chunk in _batches([record_id for record_id, _, _ in records]):
            for record_id, tag_name in session.execute(
                sa.select(AssetTag.asset_id, AssetTag.tag_name).where(AssetTag.asset_id.in_(record_chunk))
            ):
                tags.setdefault(record_id, set()).add(tag_name)
        for record_id, content_id, loader_path in records:
            roles.setdefault(content_id, []).append((loader_path, frozenset(tags.get(record_id, ()))))
    return roles


class _RoleDeriver:
    """The tags and loader path a path would be catalogued with. Both depend only on
    the folder and the extension, so they are derived once per pair."""

    def __init__(self) -> None:
        self._by_folder: dict[tuple[str, str], tuple[frozenset[str], str | None] | None] = {}

    def __call__(self, path: str) -> tuple[frozenset[str], str | None] | None:
        folder, name = os.path.split(path)
        ext = os.path.splitext(name)[1].lower()
        key = (folder, ext)
        if key not in self._by_folder:
            probe = os.path.join(folder, "x" + ext)
            try:
                tags = frozenset(get_path_derived_tags_from_path(probe))
                probe_loader = compute_loader_path(probe)
            except ValueError:
                self._by_folder[key] = None
            else:
                # The loader path ends in the file name; keep what precedes it.
                stem = None if probe_loader is None else probe_loader[: -len("x" + ext)]
                self._by_folder[key] = (tags, stem)
        derived = self._by_folder[key]
        if derived is None:
            return None
        tags, stem = derived
        return tags, None if stem is None else stem + name

    def fits(self, path: str, records: Sequence[_RecordRole]) -> bool:
        """Whether the row's records would be catalogued at ``path`` as they already
        are: the same loader path, and every tag the path implies."""
        derived = self(path)
        if derived is None:
            return False
        tags, loader_path = derived
        return all(
            (stored is None or stored == loader_path) and tags <= stored_tags
            for stored, stored_tags in records
        )


def _live_occupants(session: Session, paths: Sequence[str]) -> dict[str, PruneRow]:
    occupants: dict[str, PruneRow] = {}
    for chunk in _batches(list(paths)):
        # "is_missing = 0", as the partial unique index is declared: "IS 0" would not use it.
        stmt = sa.select(AssetContent.id, AssetContent.path, AssetContent.created_at).where(
            sa.not_(AssetContent.is_missing), AssetContent.path.in_(chunk)
        )
        for content_id, path, created_at in session.execute(stmt):
            occupants[path] = PruneRow(content_id, path, created_at)
    return occupants


def _rewrite_paths(session: Session, rewrites: Sequence[dict[str, str]]) -> int:
    """Apply path rewrites, skipping any whose path another writer took after the
    occupants were read; such a row keeps its old path until the next startup."""
    written = 0
    for chunk in _batches(rewrites):
        try:
            with session.begin_nested():
                session.execute(sa.update(AssetContent), list(chunk))
            written += len(chunk)
            continue
        except IntegrityError:
            pass
        for rewrite in chunk:
            try:
                with session.begin_nested():
                    session.execute(sa.update(AssetContent), [rewrite])
                written += 1
            except IntegrityError:
                logging.info("Asset prune: a re-home target was taken concurrently; retrying next startup")
    return written


def apply_prune_plan(session: Session, plan: PrunePlan) -> PruneResult:
    """Retire and re-home as planned.

    A row re-homes to its first candidate in its own role, or is retired if it has
    none. Where a new path is already held by a live row (a duplicate made under
    another spelling), or two rows move to one path, the oldest row keeps it, since
    it carries the user's history, and the others are retired: marked missing,
    never deleted.
    """
    retire = list(plan.retire)
    still_present = plan.still_present
    roles = _record_roles(session, [move.row.content_id for move in plan.moves if move.check_role])
    role_of = _RoleDeriver()
    by_target: dict[str, list[PruneRow]] = {}
    for move in plan.moves:
        if move.check_role:
            records = roles.get(move.row.content_id, ())
            target = next((path for path in move.candidates if role_of.fits(path, records)), None)
        else:
            target = move.candidates[0]
        if target is None:
            retire.append(move.row.content_id)
            still_present += 1
            continue
        by_target.setdefault(os.path.abspath(target), []).append(move.row)

    occupants = _live_occupants(session, list(by_target))
    rewrites: list[dict[str, str]] = []
    losers: list[str] = []
    for new_path, rows in by_target.items():
        occupant = occupants.get(new_path)
        contenders = rows if occupant is None else [*rows, occupant]
        winner = min(contenders, key=lambda row: (row.created_at, row.content_id))
        losers.extend(row.content_id for row in contenders if row is not winner)
        if winner is not occupant:
            rewrites.append({"id": winner.content_id, "path": new_path})

    # Retire first: a loser may hold the path its winner is about to take.
    for content_id in [*retire, *losers]:
        mark_content_missing(session, content_id)
    rehomed = _rewrite_paths(session, rewrites)
    session.flush()
    return PruneResult(
        marked=len(retire) + len(losers),
        rehomed=rehomed,
        still_present=still_present,
        conflict_retired=len(losers),
    )
