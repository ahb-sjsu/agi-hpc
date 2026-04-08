# AGI-HPC Project - Copyright (c) 2025-2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Tier -1: Live World Data — real-time context injection.

Only fires when the query actually needs live data. Returns empty string
for factual/knowledge questions that should use RAG instead.

Supported triggers:
  - Time/date → current datetime + timezone
  - Geo       → PostGIS lookup (countries, cities, distances)
  - Status    → Atlas hardware/service health
"""

import datetime
import re
import subprocess


def get_live_context(query):
    """Return live context string if query needs it, empty string otherwise."""
    lower = query.lower()
    parts = []

    # Time/date — only if explicitly asked
    time_triggers = [
        "what time",
        "what day",
        "what date",
        "what year",
        "current time",
        "current date",
        "today",
        "right now",
        "what month",
    ]
    if any(t in lower for t in time_triggers):
        now = datetime.datetime.now()
        utc = datetime.datetime.utcnow()
        parts.append(
            f"Current time: {now.strftime('%A, %B %d, %Y %I:%M %p')} (Pacific), "
            f"{utc.strftime('%Y-%m-%d %H:%M')} UTC."
        )

    # Geo — only if asking about places
    geo_triggers = [
        "where is",
        "capital of",
        "population of",
        "how far",
        "distance between",
        "nearest city",
        "cities in",
        "largest city",
        "smallest country",
        "latitude",
        "longitude",
        "timezone of",
        "borders",
        "continent",
    ]
    if any(t in lower for t in geo_triggers):
        geo = _query_postgis(query)
        if geo:
            parts.append(geo)

    # System status — only if asking about Atlas itself
    status_triggers = [
        "system status",
        "atlas status",
        "gpu temp",
        "cpu temp",
        "how hot",
        "services running",
        "your status",
        "are you ok",
        "how are you doing",
        "uptime",
    ]
    if any(t in lower for t in status_triggers):
        status = _get_system_status()
        if status:
            parts.append(status)

    if not parts:
        return ""

    return (
        "\n\n--- Live Data (real-time, authoritative) ---\n"
        + "\n".join(parts)
        + "\n--- End live data ---\n"
    )


def _query_postgis(query):
    """Query PostGIS for geographic information."""
    try:
        import psycopg2
    except ImportError:
        return ""

    lower = query.lower()
    results = []

    try:
        conn = psycopg2.connect("dbname=atlas user=claude")
        cur = conn.cursor()

        # Extract capitalized place names from query
        words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query)
        place_names = [w for w in words if len(w) > 2]

        # Country lookup (Natural Earth: admin=name, wkb_geometry)
        for name in place_names:
            cur.execute(
                "SELECT admin, iso_a3, pop_est, continent, "
                "ST_AsText(ST_Centroid(wkb_geometry)) "
                "FROM countries WHERE admin ILIKE %s OR name ILIKE %s LIMIT 1",
                (f"%{name}%", f"%{name}%"),
            )
            row = cur.fetchone()
            if row:
                pop = f"{row[2]:,.0f}" if row[2] else "unknown"
                results.append(
                    f"Country: {row[0]} ({row[1]}), pop: {pop}, "
                    f"continent: {row[3]}, centroid: {row[4]}"
                )

        # City lookup
        for name in place_names:
            cur.execute(
                "SELECT name, adm0name, pop_max, latitude, longitude, timezone "
                "FROM cities WHERE name ILIKE %s "
                "ORDER BY pop_max DESC NULLS LAST LIMIT 3",
                (f"%{name}%",),
            )
            for row in cur.fetchall():
                pop = f"{row[2]:,.0f}" if row[2] else "unknown"
                results.append(
                    f"City: {row[0]}, {row[1]}, pop: {pop}, "
                    f"coords: ({row[3]}, {row[4]}), tz: {row[5]}"
                )

        # Distance queries
        if "distance" in lower or "how far" in lower:
            if len(place_names) >= 2:
                cur.execute(
                    "SELECT a.name, b.name, "
                    "ST_Distance(a.wkb_geometry::geography, "
                    "b.wkb_geometry::geography)/1000 "
                    "FROM cities a, cities b "
                    "WHERE a.name ILIKE %s AND b.name ILIKE %s "
                    "ORDER BY a.pop_max DESC, b.pop_max DESC LIMIT 1",
                    (f"%{place_names[0]}%", f"%{place_names[1]}%"),
                )
                row = cur.fetchone()
                if row:
                    results.append(f"Distance: {row[0]} to {row[1]} = {row[2]:,.0f} km")

        # "cities in" queries
        if "cities in" in lower:
            for name in place_names:
                cur.execute(
                    "SELECT c.name, c.pop_max FROM cities c "
                    "WHERE c.adm0name ILIKE %s "
                    "ORDER BY c.pop_max DESC NULLS LAST LIMIT 10",
                    (f"%{name}%",),
                )
                rows = cur.fetchall()
                if rows:
                    cities_list = ", ".join(
                        f"{r[0]} ({r[1]:,.0f})" if r[1] else r[0] for r in rows
                    )
                    results.append(f"Cities in {name}: {cities_list}")

        cur.close()
        conn.close()
    except Exception as e:
        results.append(f"(geo error: {e})")

    return " | ".join(results) if results else ""


def _get_system_status():
    """Atlas system status."""
    parts = []
    try:
        out = subprocess.check_output(
            ["sensors"], text=True, timeout=3, stderr=subprocess.DEVNULL
        )
        temps = []
        for line in out.splitlines():
            if "Package" in line:
                m = re.search(r"\+(\d+\.?\d*)", line)
                if m:
                    temps.append(f"{float(m.group(1)):.0f}")
        if temps:
            parts.append(f"CPU: {'/'.join(temps)} C")

        out2 = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        for line in out2.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            parts.append(f"GPU{p[0]}: {p[1]}C, {p[2]}/{p[3]}MB")

        out3 = subprocess.check_output(["uptime", "-p"], text=True, timeout=3)
        parts.append(out3.strip())

    except Exception:
        pass
    return " | ".join(parts)
