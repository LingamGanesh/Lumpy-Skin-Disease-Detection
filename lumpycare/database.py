"""SQLite persistence layer for LumpyCare scans, cattle and demo analytics."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("LUMPYCARE_DB_PATH", str(ROOT / "data" / "lumpycare.db")))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS cattle (
                cattle_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age_years REAL,
                breed TEXT,
                village TEXT,
                latitude REAL,
                longitude REAL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                image_name TEXT NOT NULL DEFAULT '',
                prediction TEXT NOT NULL CHECK (prediction IN ('Lumpy skin','Normal skin')),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                cattle_id TEXT,
                location TEXT,
                latitude REAL,
                longitude REAL,
                source TEXT NOT NULL DEFAULT 'app',
                is_demo INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (cattle_id) REFERENCES cattle(cattle_id)
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                infection_rate REAL NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0,
                is_demo INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_scans_scanned_at ON scans(scanned_at);
            CREATE INDEX IF NOT EXISTS idx_scans_prediction ON scans(prediction);
            CREATE INDEX IF NOT EXISTS idx_scans_location ON scans(location);
            """
        )


def create_cattle(
    cattle_id: str, name: str, age_years: float | None = None, breed: str = "",
    village: str = "", latitude: float | None = None, longitude: float | None = None,
) -> dict[str, Any]:
    init_db()
    with connect() as con:
        con.execute(
            """INSERT INTO cattle(cattle_id,name,age_years,breed,village,latitude,longitude,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (cattle_id.strip(), name.strip(), age_years, breed.strip(), village.strip(),
             latitude, longitude, datetime.now(timezone.utc).isoformat()),
        )
    return get_cattle(cattle_id)


def get_cattle(cattle_id: str) -> dict[str, Any] | None:
    init_db()
    with connect() as con:
        row = con.execute("SELECT * FROM cattle WHERE cattle_id=?", (cattle_id,)).fetchone()
    return dict(row) if row else None


def list_cattle() -> list[dict[str, Any]]:
    init_db()
    with connect() as con:
        rows = con.execute(
            """SELECT c.*, COUNT(s.id) AS scan_count
               FROM cattle c LEFT JOIN scans s ON s.cattle_id=c.cattle_id
               GROUP BY c.cattle_id ORDER BY c.created_at DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def save_scan(
    prediction: str, confidence: float, image_name: str = "", cattle_id: str | None = None,
    location: str | None = None, latitude: float | None = None,
    longitude: float | None = None, source: str = "app", is_demo: bool = False,
    scanned_at: str | None = None,
) -> int:
    init_db()
    if prediction not in {"Lumpy skin", "Normal skin"}:
        raise ValueError("prediction must be Lumpy skin or Normal skin")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be between 0 and 1")
    stamp = scanned_at or datetime.now(timezone.utc).isoformat()
    with connect() as con:
        cursor = con.execute(
            """INSERT INTO scans(scanned_at,image_name,prediction,confidence,cattle_id,location,
               latitude,longitude,source,is_demo) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (stamp, image_name, prediction, float(confidence), cattle_id, location,
             latitude, longitude, source, int(is_demo)),
        )
        return int(cursor.lastrowid)


def _filters(
    start_date: str | None = None, end_date: str | None = None,
    result_type: str | None = None, min_confidence: float = 0,
    max_confidence: float = 1, cattle_id: str | None = None,
    location: str | None = None, include_demo: bool = True,
) -> tuple[str, list[Any]]:
    clauses, values = [], []
    if start_date:
        clauses.append("date(scanned_at) >= date(?)")
        values.append(start_date)
    if end_date:
        clauses.append("date(scanned_at) <= date(?)")
        values.append(end_date)
    if result_type in {"Lumpy skin", "Normal skin"}:
        clauses.append("prediction = ?")
        values.append(result_type)
    clauses.extend(["confidence >= ?", "confidence <= ?"])
    values.extend([min_confidence, max_confidence])
    if cattle_id:
        clauses.append("cattle_id = ?")
        values.append(cattle_id)
    if location:
        clauses.append("location LIKE ?")
        values.append(f"%{location}%")
    if not include_demo:
        clauses.append("is_demo = 0")
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), values


def analytics_stats(**filters: Any) -> dict[str, Any]:
    init_db()
    where, values = _filters(**filters)
    with connect() as con:
        row = con.execute(
            f"""SELECT COUNT(*) total_scans,
                SUM(CASE WHEN prediction='Lumpy skin' THEN 1 ELSE 0 END) infected,
                SUM(CASE WHEN prediction='Normal skin' THEN 1 ELSE 0 END) healthy,
                AVG(confidence) avg_confidence,
                SUM(CASE WHEN date(scanned_at)=date('now') THEN 1 ELSE 0 END) scans_today,
                SUM(CASE WHEN scanned_at >= datetime('now','-7 days') THEN 1 ELSE 0 END) scans_week,
                SUM(is_demo) demo_scans FROM scans{where}""", values
        ).fetchone()
    total = int(row["total_scans"] or 0)
    infected = int(row["infected"] or 0)
    return {
        "total_scans": total, "infected": infected, "healthy": int(row["healthy"] or 0),
        "infection_rate": round(infected / total, 4) if total else 0.0,
        "average_confidence": round(float(row["avg_confidence"] or 0), 4),
        "scans_today": int(row["scans_today"] or 0),
        "scans_week": int(row["scans_week"] or 0),
        "demo_scans": int(row["demo_scans"] or 0),
        "data_notice": "Includes synthetic demo records" if row["demo_scans"] else None,
    }


def analytics_trends(**filters: Any) -> list[dict[str, Any]]:
    init_db()
    where, values = _filters(**filters)
    with connect() as con:
        rows = con.execute(
            f"""SELECT date(scanned_at) day, COUNT(*) scans,
                SUM(CASE WHEN prediction='Lumpy skin' THEN 1 ELSE 0 END) infected,
                SUM(CASE WHEN prediction='Normal skin' THEN 1 ELSE 0 END) healthy
                FROM scans{where} GROUP BY date(scanned_at) ORDER BY day""", values
        ).fetchall()
    return [dict(row) for row in rows]


def analytics_distribution(**filters: Any) -> dict[str, Any]:
    init_db()
    where, values = _filters(**filters)
    with connect() as con:
        classes = con.execute(
            f"SELECT prediction,COUNT(*) count FROM scans{where} GROUP BY prediction", values
        ).fetchall()
        confidence = con.execute(
            f"""SELECT CASE WHEN confidence < .5 THEN '0-49%'
                WHEN confidence < .6 THEN '50-59%' WHEN confidence < .7 THEN '60-69%'
                WHEN confidence < .8 THEN '70-79%' WHEN confidence < .9 THEN '80-89%'
                ELSE '90-100%' END bucket, COUNT(*) count
                FROM scans{where} GROUP BY bucket ORDER BY bucket""", values
        ).fetchall()
    return {"classes": [dict(r) for r in classes], "confidence": [dict(r) for r in confidence]}


def analytics_activity(limit: int = 20, **filters: Any) -> list[dict[str, Any]]:
    init_db()
    where, values = _filters(**filters)
    with connect() as con:
        rows = con.execute(
            f"""SELECT s.*, c.name cattle_name FROM scans s
                LEFT JOIN cattle c ON c.cattle_id=s.cattle_id{where}
                ORDER BY scanned_at DESC LIMIT ?""", [*values, max(1, min(limit, 100))]
        ).fetchall()
    return [dict(row) for row in rows]


def analytics_geo(**filters: Any) -> list[dict[str, Any]]:
    init_db()
    where, values = _filters(**filters)
    with connect() as con:
        rows = con.execute(
            f"""SELECT location, latitude, longitude, COUNT(*) scans,
                SUM(CASE WHEN prediction='Lumpy skin' THEN 1 ELSE 0 END) infected
                FROM scans{where} AND location IS NOT NULL AND latitude IS NOT NULL
                GROUP BY location,latitude,longitude ORDER BY infected DESC""", values
        ).fetchall()
    return [dict(row) for row in rows]


def analytics_alerts(threshold: float = .30, **filters: Any) -> list[dict[str, Any]]:
    init_db()
    where, values = _filters(**filters)
    with connect() as con:
        rows = con.execute(
            f"""SELECT location,COUNT(*) scans,
                SUM(CASE WHEN prediction='Lumpy skin' THEN 1 ELSE 0 END) infected,
                AVG(CASE WHEN prediction='Lumpy skin' THEN 1.0 ELSE 0.0 END) infection_rate,
                SUM(is_demo) demo_scans
                FROM scans{where} AND location IS NOT NULL
                GROUP BY location HAVING COUNT(*) >= 5 AND infection_rate >= ?
                ORDER BY infection_rate DESC""", [*values, threshold]
        ).fetchall()
    return [dict(row) for row in rows]


def seed_demo_data(count: int = 500, seed: int = 42) -> int:
    """Insert deterministic synthetic records once; never label them as real observations."""
    import random
    rng = random.Random(seed)
    init_db()
    with connect() as con:
        existing = con.execute("SELECT COUNT(*) FROM scans WHERE is_demo=1").fetchone()[0]
        if existing:
            return int(existing)
        now = datetime.now(timezone.utc)
        villages = [f"Demo Village {i:02d}" for i in range(1, 9)]
        cattle = []
        for i in range(1, 41):
            cid = f"DEMO-COW-{i:03d}"
            village = villages[(i - 1) % len(villages)]
            lat = 17.3850 + ((i % 7) - 3) * .035
            lon = 78.4867 + ((i % 9) - 4) * .035
            cattle.append((cid, f"Demo Cattle {i:03d}", rng.randint(1, 9),
                           rng.choice(["Gir", "Sahiwal", "Red Sindhi", "Local"]),
                           village, lat, lon, now.isoformat()))
        con.executemany(
            """INSERT INTO cattle(cattle_id,name,age_years,breed,village,latitude,longitude,created_at)
               VALUES(?,?,?,?,?,?,?,?)""", cattle
        )
        records = []
        for i in range(count):
            animal = cattle[rng.randrange(len(cattle))]
            # Higher synthetic prevalence in a few fictional villages to exercise alert UI.
            probability = .48 if animal[4] in villages[:2] else .22
            infected = rng.random() < probability
            prediction = "Lumpy skin" if infected else "Normal skin"
            confidence = rng.uniform(.62, .97)
            stamp = (now - timedelta(days=rng.randrange(90), hours=rng.randrange(24))).isoformat()
            records.append((stamp, f"demo_scan_{i+1:04d}.jpg", prediction, confidence,
                            animal[0], animal[4], animal[5], animal[6], "synthetic_demo", 1))
        con.executemany(
            """INSERT INTO scans(scanned_at,image_name,prediction,confidence,cattle_id,location,
               latitude,longitude,source,is_demo) VALUES(?,?,?,?,?,?,?,?,?,?)""", records
        )
    return count
