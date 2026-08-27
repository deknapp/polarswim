"""polarswim — pull your Polar Flow swim data into a local SQLite database.

Polar Flow computes per-length swim data (lap times, pool geometry) and shows it in
the web app, but omits it from every file export it offers — FIT, TCX and CSV all
carry heart rate and nothing else. This package reads the same private endpoint the
web app uses, normalizes the result, and loads it into a queryable local database.

The committed sample database contains the author's own training data, already
published publicly on Strava, so it carries no privacy concern.

Layers, so each can be tested on its own:

    auth    resolve the browser session credential
    client  HTTP against Flow's private API (windowing, retries, rate limiting)
    parse   turn Flow's JSON into flat rows (ISO-8601 durations, sample arrays)
    db      schema, migrations, idempotent upserts
    sync    orchestration: discover -> fetch -> parse -> load, incrementally
"""

__version__ = "0.1.0"
