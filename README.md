# polarswim

Polar Flow computes per-length swim data — lap times, pool geometry, session
structure — shows it to you in the web app, and then **omits it from every file
export it offers**. Download a swim as FIT, TCX or CSV and you get heart rate and
a timestamp. Nothing else. Every pace, speed, cadence and distance column comes
back empty.

This pulls that data out of the same private endpoint the web app uses and loads
it into a local SQLite database you can query.

```bash
python -m polarswim sync --from 2024-01-01     # backfill into ~/.polarswim/polarswim.db
python -m polarswim status                     # what's stored
python -m polarswim lengths 8432902372         # one workout, length by length
pytest -q                                      # 37 tests, no network needed
```

Standard library only. `pytest` is the sole dependency, and only for the tests.

## What it recovers

For one 47-minute masters practice, the FIT export contains 2820 rows of
`(timestamp, heart_rate)` and a single lap marker covering the whole session.
The same session through this tool:

```
   #    start     dur   polar    strokes
   1     60.0s   29.6s  OTHER    0
   2     89.6s   28.0s  OTHER    0
   3    120.0s   28.0s  OTHER    0
  ...
  61   2798.4s   20.8s  OTHER    0
```

61 lengths, each with its own start offset and duration, in a 22.86 m (25 yard)
pool — none of which survives into any file Polar will hand you.

Note the `polar` column. It is `OTHER` on every length, and `strokes` is 0.
Polar's own stroke classifier needs the wrist-motion signature a watch produces;
on an arm-worn optical sensor it has nothing to work with and gives up. **The lap
times are real; the stroke labels are not there at all.** Inferring stroke from
length timing and heart rate is the next stage of this project, and it is
deliberately kept out of the storage layer.

## How it's put together

| Module | Responsibility |
|---|---|
| `auth.py` | Resolve the browser credential. Accepts a pasted cURL command verbatim, decodes the session JWT's `exp` claim, and refuses to start a long backfill on a dead session |
| `client.py` | HTTP against Flow's private API. Walks the calendar's **100-day maximum window**, retries 5xx with backoff, enforces a request floor, and recognises the HTML login page Flow serves with a 200 when a session lapses |
| `parse.py` | Pure transformation. ISO-8601 durations (`PT1M29.6S`), per-length records, and heart rate — which arrives as a bare array plus a sampling interval rather than timestamped points |
| `db.py` | Schema, idempotent upserts, transactional loads |
| `sync.py` | The loop: discover, skip what's stored, fetch, parse, load |

The layers are split this way so everything except the network is testable, and
so a change to the parser doesn't require re-fetching anything.

## Schema

```
workouts ──┬── lengths       (workout_id, idx)   one row per pool length
           ├── hr_samples    (workout_id, t_s)   flattened from values[] + interval
           ├── raw_payloads  (workout_id)        untouched API response
           └── predictions   (workout_id, idx)   inference output, kept separate
sync_runs                                        audit trail per run
model_params                                     learned parameters, refined over time
```

Decisions worth calling out:

- **Polar's training id is the primary key.** It's stable and globally unique, so
  re-syncing any date range is naturally idempotent — no "have I seen this"
  bookkeeping table, and no duplicate rows when windows overlap.
- **Raw payloads are retained.** The credential is short-lived and the API is rate
  limited, so re-fetching to fix a parser bug is expensive. `polarswim reparse`
  rebuilds every derived table from stored payloads with no network at all.
- **Children are replaced, not merged, on re-fetch.** If Polar revises a session's
  length count, the stored rows follow it exactly rather than leaving orphans.
- **Indexes follow the three real queries**: browse by date (`start_epoch`),
  filter to swims (`sport_parent, start_epoch`), and pull one workout's lengths.
- **`hr_samples` is `WITHOUT ROWID`** — it's a pure composite-key table with
  millions of narrow rows, so the extra rowid indirection is wasted space.
- **Inference output lives in its own table.** Predictions never overwrite what
  Polar actually reported, so the classifier can be re-run and compared without
  touching observed data.

## Getting a credential

Flow has no public API for this data, and its session cookie is a ~1 hour JWT with
no documented refresh. So you supply it, once per sync:

1. Open **flow.polar.com** and log in
2. DevTools → **Network**, filter for `api/training`
3. Open any training session, right-click one of those requests → **Copy as cURL**
4. `pbpaste > ~/.polarswim/cookie.txt`

Paste the whole cURL command — `auth.py` pulls the cookie out of it. If you copy a
request from `localizations.flow.polar.com` by mistake (a separate, cookie-less
host that serves UI translations), you get a clear error saying so rather than a
confusing 401 later.

The credential is never written to the database or the repository, and
`cookie.txt` and `*.db` are gitignored.

## Scope and limits

This reads an **undocumented internal API**, which can change without notice. The
parser validates the payload shape and raises rather than silently returning empty
results, so a change surfaces as a failure instead of quietly missing data.

It requests your own data from your own account, sequentially, with a rate floor
between calls.

Pool geometry is not always populated — some sessions carry lengths with no
`poolInfo` — so `pool_length_m` and `pool_type` are nullable and consumers must
handle that.

MIT licensed.
