"""Atomic publication of uploads and per-vessel recomputation.

Key invariants
--------------
1. The original file is written to a staging directory and fsynced *before*
   any database write.  It is renamed into ``originals/<batch_id>/`` before
   the batch row is committed.
2. Everything a reader can see (raw rows, verdicts, trusted points and the
   incremental change log) is committed in one transaction.  A failed upload
   therefore rolls back cleanly and leaves the previous published track
   untouched.
3. Each vessel has a monotonic ``version`` that only advances when its
   trusted track actually changes.  Every advance is recorded as an immutable
   set of ``add``/``remove`` changes (one row in ``vessel_versions`` plus the
   ``track_changes`` rows); applying them in order reproduces the current
   track.  Recomputations that do not move the track do not mint a version,
   so an old cursor stays current.
4. Recomputing a vessel is always a full replay of every committed row for
   that MMSI across all published batches, so a late telegram can move an
   earlier point from trusted to isolated (or vice versa); the diff then
   emits the matching retractions and additions.
5. A version is the smallest page unit of the change feed: pagination never
   splits the changes of one recomputation.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import ais
from .db import write_tx

STAGING_PREFIX = ".staging-"


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class CursorError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class PublishedBatch:
    batch_id: str
    filename: str
    accepted: int
    rejected: int
    vessels: int
    rows_total: int


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: Path, conn: sqlite3.Connection):
        self.root = Path(root)
        self.originals_dir = self.root / "originals"
        self.key_path = self.root / "cursor.key"
        self.conn = conn
        self.originals_dir.mkdir(parents=True, exist_ok=True)

    # -- startup recovery -------------------------------------------------

    def cleanup_staging(self) -> None:
        """Remove staging/orphan directories left by a crash.

        Only directories never referenced by a committed batch are touched;
        published batches are immutable and always survive a restart.
        """

        for entry in self.originals_dir.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith(STAGING_PREFIX):
                shutil.rmtree(entry, ignore_errors=True)
                continue
            row = self.conn.execute(
                "SELECT 1 FROM batches WHERE id = ?", (entry.name,)
            ).fetchone()
            if row is None:
                shutil.rmtree(entry, ignore_errors=True)

    # -- publication ------------------------------------------------------

    def publish(self, filename: str, payload: bytes) -> PublishedBatch:
        batch_id = uuid.uuid4().hex
        digest = hashlib.sha256(payload).hexdigest()

        try:
            outcome = ais.parse_ais_csv(payload)
        except ais.CsvFormatError as exc:
            raise UploadError(400, str(exc)) from exc

        if not outcome.rows and not outcome.rejected:
            raise UploadError(400, "文件没有可解析的数据行")
        if not outcome.rows:
            raise UploadError(
                422,
                "文件中没有任何通过校验的记录，已整体拒绝且未影响既有航迹",
            )

        staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=self.originals_dir))
        final_dir = self.originals_dir / batch_id
        try:
            staged_file = staging / "original.csv"
            with open(staged_file, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            _fsync_dir(staging)
            os.rename(staging, final_dir)
            _fsync_dir(self.originals_dir)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(final_dir, ignore_errors=True)
            raise UploadError(503, f"原档写入失败: {exc}") from exc

        relative = str(Path("originals") / batch_id / "original.csv")
        mmsis = sorted({row.mmsi for row in outcome.rows})

        try:
            with write_tx(self.conn) as conn:
                self._recheck_batch_id(conn, batch_id)
                conn.execute(
                    """
                    INSERT INTO batches (id, filename, sha256, byte_size, uploaded_at,
                                         status, rows_total, accepted_count, rejected_count,
                                         vessel_count, original_path)
                    VALUES (?, ?, ?, ?, ?, 'published', ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        filename[:255] or "upload.csv",
                        digest,
                        len(payload),
                        _utcnow(),
                        len(outcome.rows) + len(outcome.rejected),
                        len(outcome.rows),
                        len(outcome.rejected),
                        len(mmsis),
                        relative,
                    ),
                )
                self._insert_rows(conn, batch_id, outcome.rows)
                self._insert_rejects(conn, batch_id, outcome.rejected)
                for mmsi in mmsis:
                    self._recompute_vessel(conn, mmsi, batch_id)
        except Exception as exc:
            # The batch row was rolled back; its directory is now unreferenced
            # and is removed immediately (cleanup_staging covers a hard crash).
            shutil.rmtree(final_dir, ignore_errors=True)
            if isinstance(exc, UploadError):
                raise
            raise UploadError(503, f"判定写入失败，本批次未发布: {exc}") from exc

        return PublishedBatch(
            batch_id=batch_id,
            filename=filename,
            accepted=len(outcome.rows),
            rejected=len(outcome.rejected),
            vessels=len(mmsis),
            rows_total=len(outcome.rows) + len(outcome.rejected),
        )

    def _recheck_batch_id(self, conn: sqlite3.Connection, batch_id: str) -> None:
        if conn.execute("SELECT 1 FROM batches WHERE id = ?", (batch_id,)).fetchone():
            raise sqlite3.IntegrityError(f"批次 {batch_id} 已存在")

    def _insert_rows(self, conn: sqlite3.Connection, batch_id: str, rows) -> None:
        conn.executemany(
            "INSERT INTO raw_rows (batch_id, line_no, station, mmsi, utc, lat, lon) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (batch_id, r.line_no, r.station, r.mmsi, r.utc, r.lat, r.lon)
                for r in rows
            ],
        )

    def _insert_rejects(self, conn: sqlite3.Connection, batch_id: str, rejects) -> None:
        conn.executemany(
            "INSERT INTO rejects (batch_id, line_no, mmsi, station, utc, reason, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    batch_id,
                    item["line"],
                    item.get("mmsi"),
                    item.get("station"),
                    item.get("utc"),
                    item["reason"],
                    item["detail"],
                )
                for item in rejects
            ],
        )

    def _recompute_vessel(self, conn: sqlite3.Connection, mmsi: str, batch_id: str) -> None:
        rows = conn.execute(
            "SELECT id, batch_id, station, utc, lat, lon FROM raw_rows WHERE mmsi = ? "
            "ORDER BY utc, id",
            (mmsi,),
        ).fetchall()
        candidates = [
            ais.Candidate(
                row_id=r["id"],
                batch_id=r["batch_id"],
                station=r["station"],
                utc=r["utc"],
                lat=r["lat"],
                lon=r["lon"],
            )
            for r in rows
        ]
        evaluation = ais.evaluate_track(candidates)

        current = conn.execute(
            "SELECT version FROM vessels WHERE mmsi = ?", (mmsi,)
        ).fetchone()
        old_version = current["version"] if current else 0

        if current:
            old_points = conn.execute(
                "SELECT raw_row_id, batch_id, station, utc, lat, lon, speed_knots "
                "FROM track_points WHERE mmsi = ? AND version = ? ORDER BY seq",
                (mmsi, old_version),
            ).fetchall()
        else:
            old_points = []

        # Diff the freshly evaluated track against the published one.  The
        # stable identity of a point is its raw row; the ordered position is
        # the new seq so a client can keep its own ordered list correct.
        old_by_row = {p["raw_row_id"]: p for p in old_points}
        new_by_row = {p.row_id: p for p in evaluation.trusted_points}
        removed = [old_by_row[k] for k in old_by_row.keys() - new_by_row.keys()]
        added = [
            evaluation.trusted_points[i]
            for i, p in enumerate(evaluation.trusted_points)
            if p.row_id not in old_by_row
        ]
        # Same row kept but its derived speed (or ordering) shifted: retract
        # the old point and re-add the new one so clients need no patching.
        for row_id in old_by_row.keys() & new_by_row.keys():
            old_p, new_p = old_by_row[row_id], new_by_row[row_id]
            if (
                old_p["speed_knots"] != new_p.speed_knots
                or old_p["station"] != new_p.station
                or old_p["lat"] != new_p.lat
                or old_p["lon"] != new_p.lon
            ):
                removed.append(old_p)
                added.append(new_p)

        if not removed and not added and current:
            # The replay changed nothing visible (e.g. a duplicate re-send).
            # Do not mint a version: the change feed has no empty revisions and
            # every existing cursor remains exactly current.
            return

        version = old_version + 1
        now = _utcnow()

        conn.execute(
            """
            INSERT INTO vessels (mmsi, version, row_count, trusted_count, isolated_count,
                                 first_utc, last_utc, updated_at, last_batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mmsi) DO UPDATE SET
                version = excluded.version,
                row_count = excluded.row_count,
                trusted_count = excluded.trusted_count,
                isolated_count = excluded.isolated_count,
                first_utc = excluded.first_utc,
                last_utc = excluded.last_utc,
                updated_at = excluded.updated_at,
                last_batch_id = excluded.last_batch_id
            """,
            (
                mmsi,
                version,
                len(candidates),
                len(evaluation.trusted_points),
                sum(1 for c in candidates if c.status == "isolated"),
                evaluation.trusted_points[0].utc if evaluation.trusted_points else None,
                evaluation.trusted_points[-1].utc if evaluation.trusted_points else None,
                now,
                batch_id,
            ),
        )

        conn.execute(
            "DELETE FROM track_points WHERE mmsi = ? AND version = ?", (mmsi, version)
        )
        conn.execute("DELETE FROM verdicts WHERE mmsi = ? AND version = ?", (mmsi, version))

        conn.executemany(
            "INSERT INTO track_points (mmsi, version, seq, raw_row_id, batch_id, station,"
            " utc, lat, lon, speed_knots) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    mmsi,
                    version,
                    seq,
                    point.row_id,
                    point.batch_id,
                    point.station,
                    point.utc,
                    point.lat,
                    point.lon,
                    point.speed_knots,
                )
                for seq, point in enumerate(evaluation.trusted_points)
            ],
        )

        conn.executemany(
            "INSERT INTO verdicts (mmsi, version, raw_row_id, status, reason, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (mmsi, version, row_id, status, reason, detail)
                for row_id, (status, reason, detail) in evaluation.candidate_status.items()
            ],
        )

        # Immutable change feed for incremental sync.  Retractions are logged
        # first (ordered by their old position), then additions in their new
        # position; every row carries the seq it has in the resulting track so
        # one version's changes are self-describing and never page-split.
        change_rows = []
        for p in sorted(removed, key=lambda r: r["utc"]):
            change_rows.append(
                (
                    mmsi,
                    version,
                    len(change_rows),
                    "remove",
                    p["utc"],
                    p["lat"],
                    p["lon"],
                    p["speed_knots"],
                    p["station"],
                    batch_id,
                    p["batch_id"],
                    p["raw_row_id"],
                )
            )
        new_seq = {point.row_id: seq for seq, point in enumerate(evaluation.trusted_points)}
        for point in sorted(added, key=lambda c: new_seq[c.row_id]):
            change_rows.append(
                (
                    mmsi,
                    version,
                    len(change_rows),
                    "add",
                    point.utc,
                    point.lat,
                    point.lon,
                    point.speed_knots,
                    point.station,
                    batch_id,
                    point.batch_id,
                    point.row_id,
                )
            )
        conn.executemany(
            "INSERT INTO track_changes (mmsi, version, seq, op, utc, lat, lon,"
            " speed_knots, station, batch_id, point_batch_id, raw_row_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            change_rows,
        )
        conn.execute(
            "INSERT INTO vessel_versions (mmsi, version, batch_id, change_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (mmsi, version, batch_id, len(change_rows), now),
        )

        # Drop all superseded versions; readers are pinned by version and any
        # in-flight read is on a snapshot that still sees them.  The change
        # log is deliberately retained: it is the offline library's only way
        # to avoid re-downloading a whole vessel after retractions.
        conn.execute(
            "DELETE FROM track_points WHERE mmsi = ? AND version < ?", (mmsi, version)
        )
        conn.execute("DELETE FROM verdicts WHERE mmsi = ? AND version < ?", (mmsi, version))

    # -- incremental sync -------------------------------------------------

    def encode_cursor(self, mmsi: str, version: int) -> str:
        """Opaque, content-bound cursor: ``<mmsi>:<version>`` signed by HMAC.

        The token is self-describing (no server-side lookup needed to read)
        but its signature prevents a client from forging an arbitrary
        position.  Every issued token is additionally recorded in
        ``sync_cursors`` so its provenance survives a restart.
        """

        payload = f"{mmsi}:{version}".encode("utf-8")
        mac = hashlib.blake2b(payload, digest_size=16, key=self._cursor_key()).digest()
        return (
            base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
            + "."
            + base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")
        )

    def decode_cursor(self, token: str) -> tuple[str, int]:
        try:
            payload_b64, mac_b64 = token.split(".", 1)
            payload = base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4))
            mac = base64.urlsafe_b64decode(mac_b64 + "=" * (-len(mac_b64) % 4))
        except (ValueError, TypeError) as exc:
            raise CursorError(400, "游标格式无法解析") from exc
        expected = hashlib.blake2b(payload, digest_size=16, key=self._cursor_key()).digest()
        if not secrets.compare_digest(mac, expected):
            raise CursorError(400, "游标签名无效")
        try:
            mmsi, version_text = payload.decode("ascii").rsplit(":", 1)
            version = int(version_text)
        except (UnicodeDecodeError, ValueError) as exc:
            raise CursorError(400, "游标内容无法解析") from exc
        if version < 0 or not mmsi:
            raise CursorError(400, "游标内容非法")
        return mmsi, version

    def issue_cursor(self, conn: sqlite3.Connection, mmsi: str) -> tuple[str, int]:
        """Mint (or reuse) a cursor bound to the currently published track."""

        vessel = conn.execute(
            "SELECT version FROM vessels WHERE mmsi = ?", (mmsi,)
        ).fetchone()
        if vessel is None:
            raise CursorError(404, "没有该船的已发布数据")
        version = vessel["version"]
        token = self.encode_cursor(mmsi, version)
        # Recorded inside the caller's read snapshot; re-issuing the same
        # (mmsi, version) reuses the row instead of growing the table.
        conn.execute(
            "INSERT INTO sync_cursors (token, mmsi, version, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(token) DO NOTHING",
            (token, mmsi, version, _utcnow()),
        )
        return token, version

    def resolve_cursor(self, conn: sqlite3.Connection, token: str) -> tuple[str, int]:
        mmsi, version = self.decode_cursor(token)
        vessel = conn.execute(
            "SELECT version FROM vessels WHERE mmsi = ?", (mmsi,)
        ).fetchone()
        if vessel is None:
            raise CursorError(404, "游标对应的船舶不存在")
        if version > vessel["version"]:
            raise CursorError(409, "游标来自更新的数据副本，本库无法续读")
        # Remember tokens that were minted before a restart/restore too.
        conn.execute(
            "INSERT INTO sync_cursors (token, mmsi, version, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(token) DO NOTHING",
            (token, mmsi, version, _utcnow()),
        )
        return mmsi, version

    def read_changes(
        self, conn: sqlite3.Connection, mmsi: str, since_version: int, max_versions: int
    ) -> dict:
        """Return whole vessel versions strictly newer than ``since_version``.

        Pagination works on version boundaries only: the changes of one
        recomputation are always returned together, in (version, seq) order,
        so applying them in order is sufficient and the feed can never be
        half-applied.
        """

        vessel = conn.execute(
            "SELECT version FROM vessels WHERE mmsi = ?", (mmsi,)
        ).fetchone()
        if vessel is None:
            raise CursorError(404, "没有该船的已发布数据")
        current_version = vessel["version"]
        if since_version > current_version:
            raise CursorError(409, "游标来自更新的数据副本，本库无法续读")

        versions = conn.execute(
            "SELECT vv.*, b.filename AS batch_filename, b.sha256 AS batch_sha256 "
            "FROM vessel_versions vv JOIN batches b ON b.id = vv.batch_id "
            "WHERE vv.mmsi = ? AND vv.version > ? ORDER BY vv.version LIMIT ?",
            (mmsi, since_version, max_versions),
        ).fetchall()

        revisions = []
        last_version = since_version
        for vv in versions:
            changes = conn.execute(
                "SELECT * FROM track_changes WHERE mmsi = ? AND version = ? "
                "ORDER BY seq",
                (mmsi, vv["version"]),
            ).fetchall()
            revisions.append(
                {
                    "version": vv["version"],
                    "batch_id": vv["batch_id"],
                    "batch_filename": vv["batch_filename"],
                    "batch_sha256": vv["batch_sha256"],
                    "created_at": vv["created_at"],
                    "changes": [dict(c) for c in changes],
                }
            )
            last_version = vv["version"]

        return {
            "mmsi": mmsi,
            "current_version": current_version,
            "from_version": since_version,
            "last_version": last_version,
            "has_more": last_version < current_version,
            "revisions": revisions,
            "next_cursor": self.encode_cursor(mmsi, last_version),
        }

    def _cursor_key(self) -> bytes:
        """Process-stable HMAC key, persisted next to the database.

        Cursors remain verifiable across restarts because the key is a file
        with 0600 permissions created on first use; losing it only invalidates
        previously issued opaque tokens, never the change log itself.
        """

        if self.key_path.is_file():
            return self.key_path.read_bytes()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        try:
            fd = os.open(str(self.key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # concurrent first use in another worker
            return self.key_path.read_bytes()
        try:
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        return key


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
