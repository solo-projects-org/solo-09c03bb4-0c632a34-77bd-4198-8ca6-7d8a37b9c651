"""Incremental-sync tests: cursors, retractions, atomic version pages.

The offline library is modelled as a simple ``{raw_row_id: point}`` map that
is initialised from a full download and then updated by replaying the change
feed.  After every scenario the reconstructed map must equal the live track.
"""

from __future__ import annotations

import csv
import io
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as mainmod

BASE = 1767225600
HEADER = "station,mmsi,utc,lat,lon\n"
MMSI = "413000001"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(mainmod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(mainmod, "DB_PATH", tmp_path / "data" / "ais.sqlite3")
    mainmod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TestClient(mainmod.app) as c:
        c.tmp_path = tmp_path
        yield c


def upload(client, name, body: bytes | str):
    payload = body.encode() if isinstance(body, str) else body
    return client.post(
        "/upload",
        files={"file": (name, io.BytesIO(payload), "text/csv")},
        follow_redirects=False,
    )


def csv_text(*rows: str) -> str:
    return HEADER + "".join(r + "\n" for r in rows)


def parse_csv(resp):
    return list(csv.DictReader(io.StringIO(resp.text)))


def track_rows(client, mmsi=MMSI):
    return parse_csv(client.get(f"/vessels/{mmsi}/track.csv"))


def snapshot_map(rows):
    """The offline library state keyed by the stable raw-row identity."""

    return {r["raw_row_id"] if "raw_row_id" in r else r.get("utc_seconds"): r for r in rows}


def replay(client, cursor, mmsi=MMSI, limit=1000):
    """Model an offline library: seed from the full download at ``cursor``,
    then apply every page of the change feed.  Returns (state, pages, cursor).
    """

    seed = track_rows(client, mmsi)
    state = {r["utc_seconds"]: r for r in seed}
    pages = 0
    while True:
        resp = client.get(f"/vessels/{mmsi}/sync", params={"cursor": cursor, "limit": limit})
        assert resp.status_code == 200, resp.text
        pages += 1
        for change in parse_csv(resp):
            if change["op"] == "add":
                state[change["utc_seconds"]] = change
            else:
                assert change["op"] == "remove"
                state.pop(change["utc_seconds"], None)
        cursor = resp.headers["x-sync-next-cursor"]
        if resp.headers["x-sync-has-more"] == "0":
            break
    return state, pages, cursor


def test_full_download_returns_content_bound_cursor(client):
    upload(client, "b1.csv", csv_text(f"SZ01,{MMSI},{BASE},22.5,114.2"))
    resp = client.get(f"/vessels/{MMSI}/track.csv")
    assert resp.status_code == 200
    cursor = resp.headers["x-sync-cursor"]
    assert cursor and "." in cursor
    assert resp.headers["x-track-version"] == "1"
    assert f"cursor={cursor}" in resp.headers["link"]

    # The cursor endpoint hands out the same token for the same content.
    again = client.get(f"/vessels/{MMSI}/cursor").json()
    assert again["cursor"] == cursor and again["version"] == 1

    # No changes yet: an immediate sync is an empty feed.
    sync = client.get(f"/vessels/{MMSI}/sync", params={"cursor": cursor})
    assert sync.status_code == 200
    assert parse_csv(sync) == []
    assert sync.headers["x-sync-has-more"] == "0"
    assert sync.headers["x-sync-next-cursor"] == cursor


def test_initial_sync_without_cursor_rebuilds_track(client):
    upload(
        client,
        "b1.csv",
        csv_text(
            f"SZ01,{MMSI},{BASE},22.5,114.2",
            f"SZ01,{MMSI},{BASE + 60},22.505,114.205",
        ),
    )
    resp = client.get(f"/vessels/{MMSI}/sync", params={"limit": 1000})
    changes = parse_csv(resp)
    assert {c["op"] for c in changes} == {"add"}
    assert len(changes) == 2
    live = {r["utc_seconds"] for r in track_rows(client)}
    assert {c["utc_seconds"] for c in changes} == live
    # Every change cites its source batch and the batch digest.
    assert all(c["source_batch_id"] and c["source_batch_sha256"] for c in changes)


def test_late_retraction_is_delivered_as_remove_then_add(client):
    upload(
        client,
        "base.csv",
        csv_text(
            f"SZ01,{MMSI},{BASE},22.0,114.0",
            f"SZ01,{MMSI},{BASE + 60},22.01,114.01",
            f"SZ01,{MMSI},{BASE + 120},22.02,114.02",
        ),
    )
    cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]
    before = {r["utc_seconds"] for r in track_rows(client)}
    assert before == {str(BASE), str(BASE + 60), str(BASE + 120)}

    # Late conflicting telegram at t+60 isolates the earlier trusted point.
    upload(client, "late.csv", csv_text(f"HK02,{MMSI},{BASE + 60},22.9,114.9"))

    resp = client.get(f"/vessels/{MMSI}/sync", params={"cursor": cursor, "limit": 1})
    changes = parse_csv(resp)
    ops = [(c["op"], c["utc_seconds"]) for c in changes]
    assert ("remove", str(BASE + 60)) in ops
    # Retraction carries the provenance of both the point and the new batch.
    removed = next(c for c in changes if c["op"] == "remove")
    assert removed["point_batch_id"] != removed["source_batch_id"]
    assert resp.headers["x-sync-from-version"] == "1"
    assert resp.headers["x-sync-last-version"] == "2"
    assert resp.headers["x-sync-has-more"] == "0"

    state, _, final_cursor = replay(client, cursor)
    live = track_rows(client)
    assert {c["utc_seconds"] for c in state.values()} == {
        r["utc_seconds"] for r in live
    } == {str(BASE), str(BASE + 120)}

    # Replaying again from the final cursor yields nothing.
    again = parse_csv(client.get(f"/vessels/{MMSI}/sync", params={"cursor": final_cursor}))
    assert again == []


def test_one_revision_is_never_split_across_pages(client):
    upload(client, "base.csv", csv_text(f"SZ01,{MMSI},{BASE},22.0,114.0"))

    # A late batch with several close, slow points that all stay trusted:
    # v2 is one revision containing multiple add changes.
    late_rows = [
        f"SZ01,{MMSI},{BASE + 60 * i},22.{i:06d},114.{i:06d}" for i in range(1, 6)
    ]
    upload(client, "late.csv", csv_text(*late_rows))

    # Walk from v0 one revision per page: every page holds exactly one version,
    # and v2's full change set is delivered atomically on a single page.
    cursor = None
    page_versions = []
    v2_change_count = None
    while True:
        params = {"limit": 1}
        if cursor is not None:
            params["cursor"] = cursor
        resp = client.get(f"/vessels/{MMSI}/sync", params=params)
        rows = parse_csv(resp)
        versions = {r["version"] for r in rows}
        assert len(versions) == 1  # whole revision on one page, never split
        version = versions.pop()
        page_versions.append(version)
        if version == "2":
            v2_change_count = len(rows)
            assert v2_change_count > 1  # a genuinely multi-change revision
        cursor = resp.headers["x-sync-next-cursor"]
        if resp.headers["x-sync-has-more"] == "0":
            break
    assert page_versions == ["1", "2"]


def test_pagination_resumes_and_converges_on_current_track(client):
    upload(
        client,
        "b1.csv",
        csv_text(
            f"SZ01,{MMSI},{BASE},22.0,114.0",
            f"SZ01,{MMSI},{BASE + 60},22.01,114.01",
        ),
    )
    c0 = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]
    for i in range(2, 6):
        upload(
            client,
            f"b{i}.csv",
            csv_text(f"SZ01,{MMSI},{BASE + 60 * i},22.0{i},114.0{i}"),
        )

    # Page one revision at a time.
    state, pages, cursor = replay(client, c0, limit=1)
    assert pages == 4
    live = track_rows(client)
    assert {c["utc_seconds"] for c in state.values()} == {
        r["utc_seconds"] for r in live
    }
    assert cursor == client.get(f"/vessels/{MMSI}/cursor").json()["cursor"]


def test_no_change_batch_does_not_mint_version_and_keeps_cursor_current(client):
    upload(
        client,
        "b1.csv",
        csv_text(
            f"SZ01,{MMSI},{BASE},22.5,114.2",
            f"HK02,{MMSI},{BASE},22.5,114.2",  # duplicate inside the batch
        ),
    )
    cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]

    # Re-send the identical observations: replay changes nothing visible.
    upload(
        client,
        "b2_duplicate.csv",
        csv_text(
            f"SZ01,{MMSI},{BASE},22.5,114.2",
            f"HK02,{MMSI},{BASE},22.5,114.2",
        ),
    )
    page = client.get(f"/vessels/{MMSI}", headers={"Accept": "text/html"})
    assert "v1" in page.text  # version never advanced

    resp = client.get(f"/vessels/{MMSI}/sync", params={"cursor": cursor})
    assert parse_csv(resp) == []
    assert resp.headers["x-sync-next-cursor"] == cursor
    assert client.get(f"/vessels/{MMSI}/cursor").json()["cursor"] == cursor


def test_failed_upload_leaves_change_log_untouched(client):
    upload(client, "good.csv", csv_text(f"SZ01,{MMSI},{BASE},22.5,114.2"))
    cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]

    bad = upload(client, "bad.csv", csv_text("SZ01,not-a-number,1,2,3"))
    assert bad.status_code == 422

    resp = client.get(f"/vessels/{MMSI}/sync", params={"cursor": cursor})
    assert parse_csv(resp) == []
    # History endpoint also stays at the single original revision.
    history = parse_csv(client.get(f"/vessels/{MMSI}/sync"))
    assert len(history) == 1 and history[0]["op"] == "add"


def test_cursor_survives_restart_and_key_persistence(client, tmp_path, monkeypatch):
    upload(
        client,
        "b1.csv",
        csv_text(
            f"SZ01,{MMSI},{BASE},22.0,114.0",
            f"SZ01,{MMSI},{BASE + 60},22.01,114.01",
        ),
    )
    cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]
    assert (mainmod.DATA_DIR / "cursor.key").is_file()

    upload(client, "b2.csv", csv_text(f"SZ01,{MMSI},{BASE + 120},22.02,114.02"))

    # Restart: brand new app lifespan on the same data directory.
    with TestClient(mainmod.app) as c2:
        resp = c2.get(f"/vessels/{MMSI}/sync", params={"cursor": cursor})
        assert resp.status_code == 200
        changes = parse_csv(resp)
        assert len(changes) == 1 and changes[0]["op"] == "add"
        assert changes[0]["utc_seconds"] == str(BASE + 120)
        # The old cursor still converges onto the live track.
        state, _, _ = replay(c2, cursor)
        live = track_rows(c2)
        assert {c["utc_seconds"] for c in state.values()} == {
            r["utc_seconds"] for r in live
        }


def test_tampered_and_foreign_cursors_are_rejected(client):
    other = "413000002"
    upload(client, "b1.csv", csv_text(f"SZ01,{MMSI},{BASE},22.5,114.2"))
    upload(client, "b2.csv", csv_text(f"SZ01,{other},{BASE},22.6,114.3"))
    cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]

    tampered = cursor[:-2] + ("aa" if cursor[-2:] != "aa" else "bb")
    assert client.get(f"/vessels/{MMSI}/sync", params={"cursor": tampered}).status_code == 400
    assert client.get(f"/vessels/{MMSI}/sync", params={"cursor": "not-a-cursor"}).status_code == 400

    # A structurally valid cursor for another vessel cannot drive this feed.
    foreign = client.get(f"/vessels/{other}/track.csv").headers["x-sync-cursor"]
    resp = client.get(f"/vessels/{MMSI}/sync", params={"cursor": foreign})
    assert resp.status_code == 400


def test_concurrent_uploads_produce_ordered_consistent_feed(client):
    upload(client, "seed.csv", csv_text(f"SZ01,{MMSI},{BASE},22.0,114.0"))
    cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]

    def worker(i):
        body = csv_text(f"SZ01,{MMSI},{BASE + 60 * i},22.0{i},114.0{i}")
        return upload(client, f"c{i}.csv", body).status_code

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    state, pages, _ = replay(client, cursor, limit=1)
    live = track_rows(client)
    # Every concurrent revision applied exactly once; no version gap or dup.
    assert {c["utc_seconds"] for c in state.values()} == {
        r["utc_seconds"] for r in live
    }
    assert len(state) == len(live) == 7


def test_sync_html_renders_revisions(client):
    upload(client, "b1.csv", csv_text(f"SZ01,{MMSI},{BASE},22.5,114.2"))
    old_cursor = client.get(f"/vessels/{MMSI}/track.csv").headers["x-sync-cursor"]
    upload(client, "b2.csv", csv_text(f"SZ01,{MMSI},{BASE + 60},22.51,114.21"))

    # The sync page driven by the old cursor shows the new revision.
    page = client.get(
        f"/vessels/{MMSI}/sync",
        params={"cursor": old_cursor},
        headers={"Accept": "text/html"},
    )
    assert page.status_code == 200
    assert "add" in page.text and "v2" in page.text

    # The vessel page always presents the entry point and the current cursor.
    current_cursor = client.get(f"/vessels/{MMSI}/cursor").json()["cursor"]
    vessel_page = client.get(f"/vessels/{MMSI}", headers={"Accept": "text/html"})
    assert "增量同步" in vessel_page.text
    assert current_cursor in vessel_page.text
    assert old_cursor != current_cursor

    # A generic client still gets CSV, not HTML.
    csv_resp = client.get(f"/vessels/{MMSI}/sync", params={"cursor": old_cursor})
    assert csv_resp.headers["content-type"].startswith("text/csv")


def test_change_feed_rebuilds_track_after_speed_retraction_scenario(client):
    # Mirror the samples/413000006 scenario: a late point retracts an old one.
    mmsi = "413000006"
    upload(
        client,
        "base.csv",
        csv_text(
            f"SZ01,{mmsi},{BASE},22.0,114.0",
            f"SZ01,{mmsi},{BASE + 3600},22.0,114.5",
        ),
    )
    cursor = client.get(f"/vessels/{mmsi}/track.csv").headers["x-sync-cursor"]
    assert len(track_rows(client, mmsi)) == 2

    upload(client, "late.csv", csv_text(f"SZ01,{mmsi},{BASE + 3599},22.0,114.0"))

    resp = client.get(f"/vessels/{mmsi}/sync", params={"cursor": cursor})
    changes = parse_csv(resp)
    # The late t+3599 point becomes trusted; the old t+3600 point now implies
    # an impossible 1-second jump and is isolated, so it is only retracted.
    assert [c["op"] for c in changes] == ["remove", "add"]
    assert next(c for c in changes if c["op"] == "remove")["utc_seconds"] == str(BASE + 3600)
    assert next(c for c in changes if c["op"] == "add")["utc_seconds"] == str(BASE + 3599)

    state, _, _ = replay(client, cursor, mmsi=mmsi)
    live = track_rows(client, mmsi)
    assert {c["utc_seconds"] for c in state.values()} == {
        r["utc_seconds"] for r in live
    } == {str(BASE), str(BASE + 3599)}


def test_speed_change_republishes_same_point_as_remove_then_add(client):
    # A late *trusted* point shifts the derived speed of a later point without
    # removing it: the feed must retract and re-add the same raw row so an
    # offline library updates its cached speed instead of keeping a stale one.
    mmsi = "413000007"
    upload(
        client,
        "base.csv",
        csv_text(
            f"SZ01,{mmsi},{BASE},22.0,114.0",
            f"SZ01,{mmsi},{BASE + 120},22.002,114.002",
        ),
    )
    cursor = client.get(f"/vessels/{mmsi}/track.csv").headers["x-sync-cursor"]
    first_speed = {r["utc_seconds"]: r["speed_knots"] for r in track_rows(client, mmsi)}

    # A close, slow point at t+60 stays trusted and becomes the new anchor.
    upload(client, "late.csv", csv_text(f"SZ01,{mmsi},{BASE + 60},22.0001,114.0001"))

    changes = parse_csv(client.get(f"/vessels/{mmsi}/sync", params={"cursor": cursor}))
    by_op = {}
    for c in changes:
        by_op.setdefault(c["op"], []).append(c)
    assert len(by_op["remove"]) == 1 and len(by_op["add"]) == 2
    republished = {c["utc_seconds"] for c in by_op["remove"]} & {
        c["utc_seconds"] for c in by_op["add"]
    }
    assert republished == {str(BASE + 120)}
    new_speed = next(
        c["speed_knots"] for c in by_op["add"] if c["utc_seconds"] == str(BASE + 120)
    )
    assert new_speed != first_speed[str(BASE + 120)]

    state, _, _ = replay(client, cursor, mmsi=mmsi)
    live = {r["utc_seconds"]: r for r in track_rows(client, mmsi)}
    assert set(state) == set(live)
    assert state[str(BASE + 120)]["speed_knots"] == live[str(BASE + 120)]["speed_knots"]


def test_sync_unknown_vessel_404_and_bad_limit(client):
    assert client.get("/vessels/999999999/sync").status_code == 404
    assert client.get("/vessels/999999999/cursor").status_code == 404
    upload(client, "b1.csv", csv_text(f"SZ01,{MMSI},{BASE},22.5,114.2"))
    assert client.get(f"/vessels/{MMSI}/sync", params={"limit": 0}).status_code == 400
