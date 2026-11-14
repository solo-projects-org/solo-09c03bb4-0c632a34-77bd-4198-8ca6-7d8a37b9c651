"""End-to-end smoke run over the bundled samples (filesystem + SQLite, no HTTP).

Usage: python -m tests.smoke            (from an activated venv)
       python tests/smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ais, db
from app.store import Store


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db.init_db(root / "ais.sqlite3")
        conn = db.connect(root / "ais.sqlite3")
        store = Store(root, conn)

        b1 = store.publish("batch1_base.csv", Path("samples/batch1_base.csv").read_bytes())
        print(f"batch1: accepted={b1.accepted} rejected={b1.rejected} vessels={b1.vessels}")

        b2 = store.publish(
            "batch2_late_arrivals.csv",
            Path("samples/batch2_late_arrivals.csv").read_bytes(),
        )
        print(f"batch2: accepted={b2.accepted} rejected={b2.rejected} vessels={b2.vessels}")

        # --- incremental sync: remember a cursor before the next publication ---
        cursor_v6 = store.encode_cursor("413000006", 2)
        live_v6_before = {
            r["utc"]: dict(r)
            for r in conn.execute(
                "SELECT utc, raw_row_id FROM track_points WHERE mmsi='413000006' AND version=2"
            )
        }

        b3 = store.publish(
            "batch3_incremental.csv",
            Path("samples/batch3_incremental.csv").read_bytes(),
        )
        print(f"batch3: accepted={b3.accepted} rejected={b3.rejected} vessels={b3.vessels}")

        # A no-op replay (identical observations) must not mint a version or
        # append any change, and must leave every cursor exactly current.
        version_before = conn.execute(
            "SELECT version FROM vessels WHERE mmsi='413000006'"
        ).fetchone()["version"]
        changes_before = conn.execute(
            "SELECT COUNT(*) c FROM track_changes WHERE mmsi='413000006'"
        ).fetchone()["c"]
        store.publish(
            "batch3_replay.csv",
            Path("samples/batch3_incremental.csv").read_bytes(),
        )
        version_after = conn.execute(
            "SELECT version FROM vessels WHERE mmsi='413000006'"
        ).fetchone()["version"]
        changes_after = conn.execute(
            "SELECT COUNT(*) c FROM track_changes WHERE mmsi='413000006'"
        ).fetchone()["c"]
        assert version_before == version_after, "no-op replay advanced a vessel version"
        assert changes_before == changes_after, "no-op replay appended changes"
        print(f"no-op replay: 413000006 stays at v{version_after} with {changes_after} changes")

        # --- read the change feed from the old cursor, one revision per page ---
        feed = store.read_changes(conn, "413000006", 2, max_versions=100)
        assert len(feed["revisions"]) == 1, "expected exactly one new revision"
        revision = feed["revisions"][0]
        ops = [(c["op"], c["utc"]) for c in revision["changes"]]
        print(f"413000006 v{revision['version']} changes: {ops}")
        assert ops == [("add", 1767229260)], "batch3 should append one trusted point"
        assert feed["has_more"] is False
        assert store.decode_cursor(feed["next_cursor"]) == ("413000006", revision["version"])

        # Replaying the feed on top of the cursor snapshot yields the live track.
        applied = dict(live_v6_before)
        for c in revision["changes"]:
            if c["op"] == "add":
                applied[c["utc"]] = c
            else:
                applied.pop(c["utc"], None)
        live_v6_after = {
            r["utc"]: dict(r)
            for r in conn.execute(
                "SELECT utc, raw_row_id FROM track_points WHERE mmsi='413000006' "
                "AND version=(SELECT version FROM vessels WHERE mmsi='413000006')"
            )
        }
        assert set(applied) == set(live_v6_after), "replayed feed does not match live track"
        print("incremental replay of 413000006 converges on the live track")

        # The old cursor is still resolvable and restarts need no special case.
        assert store.decode_cursor(cursor_v6) == ("413000006", 2)
        assert store.read_changes(conn, "413000006", 2, 1)["revisions"][0]["version"] == 3

        # A late conflicting telegram retracts an earlier trusted point: the
        # revision then carries a remove, proving the offline library learns
        # about deletions without re-downloading the whole vessel.
        store.publish(
            "batch4_conflict.csv",
            (
                "station,mmsi,utc,lat,lon\n"
                "HK02,413000009,1767225600,23.900000,115.900000\n"
            ).encode(),
        )
        conflict_feed = store.read_changes(conn, "413000009", 1, 100)
        conflict_ops = [(c["op"], c["utc"]) for c in conflict_feed["revisions"][0]["changes"]]
        print(f"413000009 conflict revision changes: {conflict_ops}")
        assert ("remove", 1767225600) in conflict_ops, "conflict retraction missing"

        # A brand-new vessel gets its own v1 revision made only of additions.
        feed_v9 = store.read_changes(conn, "413000009", 0, 100)
        assert all(c["op"] == "add" for c in feed_v9["revisions"][0]["changes"])
        print("new vessel 413000009 starts from an all-add first revision")

        flipped = conn.execute(
            "SELECT isolated_count, version FROM vessels WHERE mmsi = '413000006'"
        ).fetchone()
        isolated_416 = conn.execute(
            """
            SELECT COUNT(*) c FROM verdicts v
            WHERE v.mmsi = '413000006'
              AND v.version = (SELECT version FROM vessels WHERE mmsi = '413000006')
              AND v.status = 'isolated'
            """
        ).fetchone()["c"]
        assert isolated_416 == 1, "late point did not change the earlier verdict"
        print(
            f"late point flipped an earlier point: v{flipped['version']} isolated={isolated_416}"
        )

        for mmsi in ("413000001", "413000002", "413000003", "413000005", "413000006"):
            row = conn.execute("SELECT * FROM vessels WHERE mmsi = ?", (mmsi,)).fetchone()
            print(
                f"\nMMSI {mmsi}: v{row['version']} rows={row['row_count']} "
                f"trusted={row['trusted_count']} isolated={row['isolated_count']}"
            )
            for p in conn.execute(
                "SELECT * FROM track_points WHERE mmsi = ? AND version = ? ORDER BY seq",
                (mmsi, row["version"]),
            ):
                speed = "" if p["speed_knots"] is None else f"{p['speed_knots']:.1f} kn"
                print(f"  trusted {p['utc']} {p['lat']:.6f},{p['lon']:.6f} {speed}")
            for v in conn.execute(
                """
                SELECT r.utc, r.station, v.reason, v.detail
                FROM verdicts v JOIN raw_rows r ON r.id = v.raw_row_id
                WHERE v.mmsi = ? AND v.version = ? AND v.status = 'isolated'
                ORDER BY r.utc, r.id
                """,
                (mmsi, row["version"]),
            ):
                print(f"  isolated {v['utc']} [{v['station']}] {v['reason']}: {v['detail'] or ''}")

        # An all-bad upload is rejected wholesale and leaves tracks untouched.
        before = conn.execute(
            "SELECT version, trusted_count FROM vessels WHERE mmsi = '413000003'"
        ).fetchone()
        try:
            store.publish("bad.csv", b"station,mmsi,utc,lat,lon\nSZ01,x,1,2,3\n")
        except Exception as exc:  # noqa: BLE001 - expected
            print(f"\nall-bad upload rejected: {type(exc).__name__}: {exc}")
        after = conn.execute(
            "SELECT version, trusted_count FROM vessels WHERE mmsi = '413000003'"
        ).fetchone()
        assert tuple(before) == tuple(after), "failed upload changed a published track"
        print("published tracks unchanged after failed upload")

        # The pure evaluator agrees with the persisted state.
        rows = conn.execute(
            "SELECT id, batch_id, station, utc, lat, lon FROM raw_rows WHERE mmsi='413000003'"
        ).fetchall()
        ev = ais.evaluate_track(
            [
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
        )
        persisted = conn.execute(
            "SELECT utc FROM track_points WHERE mmsi='413000003' AND version=1 ORDER BY seq"
        ).fetchall()
        assert [p.utc for p in ev.trusted_points] == [r["utc"] for r in persisted]
        print("evaluator matches persisted track")

        conn.close()
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
