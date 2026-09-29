"""Keeps a catalogued row through the startup prune when its folder is still owned
but spelled differently today: an 8.3 short name, a junction, a subst drive, a
``\\\\?\\`` prefix, a symlink, or letter case. The prune rewrites such a row's
folder part to today's spelling instead of retiring it, so the row keeps its id,
name, tags and job links, and the scan that follows finds it at the path it lists.

Every live row then carries today's exact spelling, which is what lets the
per-root sync, the scan's existing-path dedupe and the live-path unique index
stay exact-string comparisons.

Two folders are the same only when stat says so (device and inode), and a row is
re-homed only when its file stats at the new path. Anything undecided (a stat
that fails, a filesystem reporting inode 0) is retired, as before.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import NamedTuple

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.assets.database.models import AssetContent
from app.assets.database.queries.records import mark_content_missing

_BATCH = 500

_DirIdentity = tuple[int, int]


class PruneResult(NamedTuple):
    marked: int  # rows retired, conflict losers included
    rehomed: int  # rows rewritten to today's spelling of their folder
    still_present: int  # retired rows whose file still stats
    conflict_retired: int  # retired because an older row holds the same path


class PruneRow(NamedTuple):
    content_id: str
    path: str
    created_at: datetime


class PrunePlan(NamedTuple):
    moves: list[tuple[PruneRow, str]]
    retire: list[str]
    still_present: int


class RespelledRow(NamedTuple):
    row: PruneRow
    new_path: str


class CaseRespeller:
    """Maps a path to its folder part in the spelling of the deepest prefix it lies
    under by the platform's case rules. Pure string: the prune already treats a
    normcase match as the same folder. The prefixes are normalised once, since the
    prune calls this for every live row."""

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

    def __call__(self, path: str) -> str | None:
        """``path`` respelled, or None when no prefix contains it. Deepest prefix first,
        so a row exactly under a shallow prefix is still respelled for a deeper one."""
        candidate = os.path.normcase(path)
        for prefix, folded, stem in self._stems:
            if candidate == folded or candidate.startswith(stem):
                return prefix + path[len(prefix):]
        return None


def _stat_or_none(path: str) -> os.stat_result | None:
    try:
        return os.stat(path)
    except (OSError, ValueError):
        return None


def _identity(stat_result: os.stat_result | None) -> _DirIdentity | None:
    # Inode 0 means the filesystem (some SMB and FUSE mounts) reports no file ids,
    # so every directory on it would compare equal.
    if stat_result is None or not stat.S_ISDIR(stat_result.st_mode) or stat_result.st_ino == 0:
        return None
    return (stat_result.st_dev, stat_result.st_ino)


class _FolderResolver:
    """Stats each directory at most once, so the cost is per distinct folder, not per row."""

    def __init__(self, prefixes: Sequence[str]):
        self._prefixes = [os.path.abspath(p) for p in prefixes]
        self._stats: dict[str, os.stat_result | None] = {}
        self._owned: dict[_DirIdentity, str] | None = None
        self._owners: dict[str, tuple[str, str] | None] = {}

    def dir_stat(self, directory: str) -> os.stat_result | None:
        if directory not in self._stats:
            self._stats[directory] = _stat_or_none(directory)
        return self._stats[directory]

    def _owned_identities(self) -> dict[_DirIdentity, str]:
        if self._owned is None:
            self._owned = {}
            for prefix in self._prefixes:
                identity = _identity(self.dir_stat(prefix))
                if identity is not None:
                    self._owned.setdefault(identity, prefix)
        return self._owned

    def owner(self, directory: str) -> tuple[str, str] | None:
        """(ancestor, prefix): the deepest ancestor of ``directory`` (itself
        included) that is the same folder as an owned prefix, or None."""
        owned = self._owned_identities()
        chain: list[str] = []
        current = directory
        while True:
            if current in self._owners:
                found = self._owners[current]
                break
            chain.append(current)
            identity = _identity(self.dir_stat(current))
            if identity is not None and identity in owned:
                found = (current, owned[identity])
                break
            parent = os.path.dirname(current)
            if parent == current:
                found = None
                break
            current = parent
        # No directory in the chain matched, so each one's deepest match is ``found``.
        for walked in chain:
            self._owners[walked] = found
        return found


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
    moves: list[tuple[PruneRow, str]] = [(r.row, r.new_path) for r in respelled]
    retire: list[str] = []
    still_present = 0

    resolver = _FolderResolver(prefixes)
    for row in unowned:
        parent = os.path.dirname(row.path)
        # A parent that cannot be stat'ed (an unplugged drive, an offline share) skips
        # the per-file stat, so an offline folder costs one failed stat, not one per file.
        if resolver.dir_stat(parent) is None or _stat_or_none(row.path) is None:
            retire.append(row.content_id)
            continue
        found = resolver.owner(parent)
        if found is not None:
            ancestor, prefix = found
            new_path = os.path.join(prefix, row.path[len(ancestor):].lstrip(os.sep))
            if _stat_or_none(new_path) is not None:
                moves.append((row, new_path))
                continue
        retire.append(row.content_id)
        still_present += 1
    return PrunePlan(moves, retire, still_present)


def _batches(items: Sequence, size: int = _BATCH) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def apply_prune_plan(session: Session, plan: PrunePlan) -> PruneResult:
    """Retire and re-home as planned. Where a new path is already held by a live row
    (a duplicate made under another spelling), or two rows move to one path, the
    oldest row keeps it, since it carries the user's history, and the others are
    retired: marked missing, never deleted."""
    by_target: dict[str, list[PruneRow]] = {}
    for row, new_path in plan.moves:
        by_target.setdefault(new_path, []).append(row)

    occupants: dict[str, PruneRow] = {}
    for chunk in _batches(list(by_target)):
        stmt = sa.select(AssetContent.id, AssetContent.path, AssetContent.created_at).where(
            AssetContent.is_missing.is_(False), AssetContent.path.in_(chunk)
        )
        for content_id, path, created_at in session.execute(stmt):
            occupants[path] = PruneRow(content_id, path, created_at)

    rewrites: list[dict[str, str]] = []
    losers: list[str] = []
    for new_path, rows in by_target.items():
        occupant = occupants.get(new_path)
        contenders = rows if occupant is None else [*rows, occupant]
        winner = min(contenders, key=lambda row: (row.created_at, row.content_id))
        losers.extend(row.content_id for row in contenders if row is not winner)
        if winner is not occupant:
            rewrites.append({"id": winner.content_id, "path": os.path.abspath(new_path)})

    # Retire first: a loser may hold the path its winner is about to take.
    for content_id in [*plan.retire, *losers]:
        mark_content_missing(session, content_id)
    for chunk in _batches(rewrites):
        session.execute(sa.update(AssetContent), list(chunk))
    session.flush()
    return PruneResult(
        marked=len(plan.retire) + len(losers),
        rehomed=len(rewrites),
        still_present=plan.still_present,
        conflict_retired=len(losers),
    )
