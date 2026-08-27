-- polarswim schema.
--
-- Design notes:
--   * Polar's own training id is the primary key everywhere. It is stable and
--     globally unique, so re-syncing is naturally idempotent.
--   * Raw payloads are retained in `raw_payloads` so the parse and analysis
--     layers can be changed and re-run over history without re-fetching (the
--     API is rate-limited and the session credential is short-lived).
--   * `lengths` is the fact table everything else hangs off, keyed by
--     (workout_id, idx) because a length has no identifier of its own.
--   * Indexes cover the three real access patterns: browse by date, filter to
--     pool swims, and pull one workout's lengths or samples.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS workouts (
    id                     INTEGER PRIMARY KEY,   -- Polar training id
    start_time             TEXT    NOT NULL,      -- ISO-8601, device local time
    start_epoch            INTEGER NOT NULL,      -- derived, for range scans
    stop_time              TEXT,
    sport_parent           TEXT,                  -- e.g. 'SWIMMING'
    sport_id               INTEGER,
    duration_s             REAL,
    distance_m             REAL,
    calories               INTEGER,
    avg_hr                 INTEGER,
    max_hr                 INTEGER,
    pool_length_m          REAL,                  -- NULL unless a pool swim
    pool_type              TEXT,                  -- 'YARDS' | 'METERS'
    pool_lengths_reported  INTEGER,               -- Polar's own count, for validation
    n_lengths              INTEGER NOT NULL DEFAULT 0,  -- what we actually stored
    hr_interval_s          REAL,
    n_hr_samples           INTEGER NOT NULL DEFAULT 0,
    synced_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workouts_start  ON workouts(start_epoch);
CREATE INDEX IF NOT EXISTS idx_workouts_sport  ON workouts(sport_parent, start_epoch);

-- One row per pool length, as Polar's turn detection saw it.
CREATE TABLE IF NOT EXISTS lengths (
    workout_id      INTEGER NOT NULL,
    idx             INTEGER NOT NULL,   -- 1-based, in time order within the workout
    start_offset_s  REAL    NOT NULL,   -- from workout start
    duration_s      REAL    NOT NULL,
    polar_style     TEXT,               -- Polar's claim; 'OTHER' when it cannot tell
    strokes         INTEGER,
    PRIMARY KEY (workout_id, idx),
    FOREIGN KEY (workout_id) REFERENCES workouts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lengths_duration ON lengths(duration_s);

-- Heart rate, flattened from Flow's (values[], interval) representation so it can
-- be joined to lengths on time.
CREATE TABLE IF NOT EXISTS hr_samples (
    workout_id  INTEGER NOT NULL,
    t_s         REAL    NOT NULL,       -- seconds from workout start
    hr          INTEGER NOT NULL,
    PRIMARY KEY (workout_id, t_s),
    FOREIGN KEY (workout_id) REFERENCES workouts(id) ON DELETE CASCADE
) WITHOUT ROWID;

-- Untouched API responses, so reprocessing never needs the network.
CREATE TABLE IF NOT EXISTS raw_payloads (
    workout_id  INTEGER PRIMARY KEY,
    fetched_at  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    FOREIGN KEY (workout_id) REFERENCES workouts(id) ON DELETE CASCADE
);

-- Audit trail: what each sync run covered and what it changed.
CREATE TABLE IF NOT EXISTS sync_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    window_start  TEXT,
    window_end    TEXT,
    events_seen   INTEGER NOT NULL DEFAULT 0,
    fetched       INTEGER NOT NULL DEFAULT 0,
    skipped       INTEGER NOT NULL DEFAULT 0,
    errors        INTEGER NOT NULL DEFAULT 0,
    note          TEXT
);

-- Learned per-class parameters, refined as more workouts are synced. Keeping the
-- model in the database (rather than a pickle) means it is inspectable, diffable,
-- and versioned alongside the data it was estimated from.
CREATE TABLE IF NOT EXISTS model_params (
    class       TEXT NOT NULL,   -- freestyle | backstroke | breaststroke | butterfly
                                 -- | kick | drill | undetermined
    param       TEXT NOT NULL,   -- e.g. 'duration_mean_s', 'duration_sd_s', 'n_obs'
    value       REAL NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (class, param)
);

-- Per-length inference output, kept separate from the observed data so the
-- classifier can be re-run without touching anything Polar gave us.
CREATE TABLE IF NOT EXISTS predictions (
    workout_id    INTEGER NOT NULL,
    idx           INTEGER NOT NULL,
    predicted     TEXT    NOT NULL,
    confidence    REAL,
    method        TEXT,
    predicted_at  TEXT    NOT NULL,
    PRIMARY KEY (workout_id, idx),
    FOREIGN KEY (workout_id, idx) REFERENCES lengths(workout_id, idx) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted);
