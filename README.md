# polarswim

Polar Flow computes per-length swim data — lap times, pool geometry, set structure —
shows it to you in its web app, and then **omits it from every file export it
offers**. Download a swim as FIT, TCX or CSV and you get heart rate and a timestamp.
Nothing else: speed, pace, cadence and distance columns all come back empty.

Worse, on an arm-worn sensor Polar's own stroke classifier gives up entirely. Across
7,615 real lengths it labelled **every single one `OTHER`**, with a stroke count of
zero.

This recovers the per-length data from the same private endpoint the web app uses,
loads it into a queryable database, and infers the stroke Polar couldn't.

```bash
pip install -r requirements.txt

# Everything below runs against the committed sample database — no account needed.
python -m polarswim --db sample/sample.db status
python -m polarswim --db sample/sample.db analyze
python -m polarswim --db sample/sample.db card 2026-08-19    # paste into Strava
python -m polarswim --db sample/sample.db report --from 2026-08-01
python -m polarswim --db sample/sample.db serve               # web UI on :8770

pytest -q                                                    # 211 tests, no network
```

`sample/sample.db` holds six real swims — 406 lengths and 16,810 heart-rate samples.
This is my own training data, already published publicly on Strava, so there is no
privacy concern in shipping it; no credentials, tokens, or API responses are included.

## The Strava card

`polarswim card` emits plain Unicode sized for a phone, so it pastes straight into
a Strava description:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃🏊 2026-08-19  1,525 yd  47:03   ┃
┃   61 lengths  ·  avg 126 bpm   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│  6×25 brst ▇▇▇       29s       │
│  4×25 free ▇▇▇▇      28s       │
│  3×25 free ▇▇▇       30s       │
│  3×25 drll ▇         36s       │
│ 12×25 back ▇▇▇       32s       │
│  1×25 free ▇▇▇▇▇     24s       │
│  2×25 fly  ▇▇▇▇▇▇    20s       │
│ 12×25 fly  ▇▇▇▇      28s       │
│  2×25 back ▇▇▇       31s       │
│  4×25 back ▇▇▇       30s       │
│  2×25 back ▇▇        32s       │
│  2×25 brst ▇▇▇▇▇     24s       │
│  4×25 free ▇▇▇▇      28s       │
│  3×25 free ▇▇▇▇      29s       │
│  1×25 free ▇▇▇▇▇▇    21s       │
└────────────────────────────────┘

pace by length (taller = faster)
  1 ▅▅▅▅▅▄▅▆▅▃▁█▅▂▁▇▆▄▁▄▃█▃▆▃▄▃█▇▆
 31 █▃▃▂▄▄▆▅▇▄▇▅▅▃▄▃▄▄▄▃▄▄▇▄▄▁▇▄▂█
```

The card is coloured. Strava descriptions are plain text — no markdown, no HTML, no
ANSI colour — but emoji render in colour everywhere, so the stroke mix is a
proportional stacked bar of coloured squares and every set row is tagged with its
stroke's colour. The web dashboard draws the same breakdown as a real SVG pie using
matching colours.

## Reps, not raw lengths

The sensor records one row per pool length, but that is not how a practice is
swum. Four unbroken lengths of a 25 yd pool is a **100**, and reporting it as four
25s misdescribes the session.

Consecutive lengths with no rest between them are grouped into a **rep**, and
consecutive reps of equal distance into a **set** — so the card reads `4×50`, not
`8×25`. The threshold is two seconds, chosen from the data rather than by feel: the
observed gap distribution is sharply bimodal, with 64.6% of gaps exactly zero and
only 0.5% falling between zero and two seconds, so anything from 0.5 s to 5 s gives
the same grouping.

Statistics are still computed over the **set**, because pooling every rep of the
same distance gives far more lengths to estimate a median and a spread from.

## Swimmer-calibrated metrics

The dashboard adds three columns, each referenced to this swimmer rather than a
population table:

**Heart-rate zone.** Anchored to the highest heart rate actually observed *while
swimming* (172 bpm here). Maximum heart rate in water runs roughly 10–13 bpm below a
land-based maximum — horizontal position, cooling water, less working muscle — so an
age formula or a running max would place every boundary too high. Colour-coded, with
the bpm ranges shown as a key.

**Relative speed.** A rep's percentile against this swimmer's own reps of the **same
distance**. Ranking within an inferred stroke was tried and rejected as circular: the
classifier assigns fast lengths to freestyle, so a freestyle percentile mostly
re-expresses the classifier's threshold, and it collapsed every slow length to 0%.
Distance is measured rather than inferred, so it carries no such feedback loop.
A distance with fewer than 30 reps reports nothing instead of a fragile number.

**Personal best.** The fastest recorded time at a distance and stroke. Two caveats
are built in: the stroke is inferred, so a best is provisional; and reps faster than
65% of the swimmer's median pace are excluded as turn-detection artifacts — before
that filter the "best" 25 yd freestyle was 13.6 s against a 26 s median, which is a
split length, not a swim. 45 such reps were excluded.

## Uploading to Strava

Two outputs, because Strava takes two kinds of content:

**The text card** (`polarswim card <date>`) pastes into an activity description.
Plain text with coloured emoji, since that is all a description renders.

**The image** — the *download image for Strava* button in the dashboard — is the
full analysis as a PNG you can attach to the activity as a photo: stroke-mix donut,
per-set table with heart-rate zones, relative speed and personal bests, and the zone
key. It is built server-side as SVG and rasterised in the browser through a canvas,
so it needs no plotting library and no external script. The SVG deliberately avoids
`foreignObject` and any external reference, both of which taint a canvas and would
make the export fail silently; a test enforces that.

## Effort: two numbers, not one

"How hard was that?" is two questions, and one number cannot answer both.

**Load** is accumulated stress, so it grows with duration — a three-hour swim
outranks a sharp hour, correctly, because it is more total work. **Intensity** is
load per minute, duration-independent, and answers how hard it was while it lasted.
On this database the two rank sessions completely differently: load picks the
three-hour swims, intensity picks the ~1-hour sessions at 66% heart-rate reserve.
Both are percentiles against the swimmer's own history, so they stay meaningful as
fitness changes.

The load model is **Banister TRIMP**: every second weighted by heart-rate reserve
through `x · 0.64 · e^(1.92·x)`. The exponential is the point. A plain integral of
heart rate is linear, so it scores 30 minutes easy and 15 minutes hard about
equally; Edwards' zone weights (1–5) are better but still linear. Under Banister,
30 minutes at 155 bpm outweighs two hours at 115 bpm — which matches how those two
sessions actually feel.

## Naming a workout

Commands that act on one session take a **date**, a Polar training id, or `latest` —
nobody remembers a training id:

```
python -m polarswim card 2026-08-19
python -m polarswim card latest
python -m polarswim review 2026-08-19
```

Two swims on the same day makes the date ambiguous, so it lists the candidates
instead of silently picking one.

## Web UI

```
python -m polarswim serve
```

Then open **http://127.0.0.1:8770**. Swims are listed newest first; picking one shows
a pace-per-length chart, a set-by-set table with **low-confidence stroke labels
highlighted** so an estimate never reads as a measurement, a one-click copy of the
Strava card, and the AI review. The date pickers run the season report over any
range. It binds to localhost only — this is a personal tool over a personal
database, not a service.

## Inferring stroke without labels

There is no ground truth here — the vendor labelled nothing, and hand-labelling
years of practices is not realistic. So the classifier leans on structure instead:

**Sets.** Rests split a practice into sets. Within a set the swimmer is doing one
thing, which gives every length a local reference that adapts to that day's effort.

**Repair before classify.** Polar's turn detection misses walls, fusing two lengths
into one record. A slow length is ambiguous alone — merged pair, or genuinely slow
drill? — but not in context. A merge is an **isolated near-integer multiple** of its
set's median (2.0x, 3.9x); a drill set is **uniformly** slow (1.0–1.4x). Across the
full dataset that split 77 slow lengths into 38 merges and 39 real drills. Repaired
boundaries are marked `inferred_split` in the database and never silently mixed in
with measured data.

**Two axes, no assumed ordering.** Per length we derive normalized pace (seconds per
25 yd, so a 50 m pool is comparable) and heart-rate cost above that workout's own
baseline. Both matter, because per-swimmer speed order is *not* universal — plenty of
swimmers are slower at backstroke than breaststroke. Nothing here assumes a ranking:

| | pace | cost | reasoning |
|---|---|---|---|
| freestyle | fast | any | the dominant mode, and the default hypothesis |
| butterfly | mid | **high** | expensive for the speed it buys |
| breaststroke | slow | **low** | the glide phase makes it cheap |
| backstroke | slow | **high** | working hard without travelling |
| other | slow, uniform set | low | drill and kick share one class |

Breaststroke and a weak backstroke are indistinguishable on pace and sit in opposite
corners on cost — which is the whole reason for the second axis.

**It says "undetermined".** Where the evidence doesn't separate two classes, that is
the answer. On the full dataset 6% of lengths come back undetermined rather than
being assigned a coin-flip label.

**It learns.** Reference paces are estimated from the swimmer's own history and
written to `model_params`, so they tighten as workouts are synced. Keeping the model
in the database rather than a pickle makes it inspectable and diffable, and it is
where ground-truth labels would pin the clusters if any were ever supplied.

## Architecture

| Module | Responsibility |
|---|---|
| `auth` | Credential from a pasted cURL; decodes the session JWT's `exp` and refuses to start a long backfill on a dead session |
| `client` | Flow's private API — walks the calendar's **100-day cap**, retries 5xx, rate limits, and catches the HTML login page Flow serves with a 200 when a session lapses |
| `parse` | Pure transformation: ISO-8601 durations, per-length records, HR arrays |
| `models` | Schema as SQLAlchemy Core tables |
| `db` | Idempotent upserts, transactional loads |
| `sync` | Discover, skip stored, fetch, parse, load |
| `analyze` | Sets, merge repair, features, classification, learned parameters |
| `render` | Unicode cards |
| `report` | pandas aggregation over a date range |
| `ai` | Optional Claude review of one session |
| `web` | Local Flask UI |
| `spark` | Optional PySpark path (see below) |

## Schema

```
workouts ──┬── lengths       (workout_id, idx)   one row per pool length
           ├── hr_samples    (workout_id, t_s)   flattened from values[] + interval
           ├── raw_payloads  (workout_id)        untouched API response
           └── predictions   (workout_id, idx)   inference, kept apart from observation
sync_runs                                        audit trail per run
model_params                                     learned parameters, refined over time
```

- **Polar's training id is the primary key**, so re-syncing any range is naturally
  idempotent — no "have I seen this" table, no duplicates on overlapping windows.
- **Raw payloads are retained.** The credential is short-lived and the API rate
  limited, so `reparse` rebuilds every derived table with no network at all.
- **Children are replaced, not merged, on re-fetch** — no orphans if Polar revises
  a session's length count.
- **Indexes follow the real queries**: browse by date, filter to swims, pull one
  workout's children.
- **`hr_samples` is `WITHOUT ROWID`** — a narrow composite-key table with hundreds
  of thousands of rows, where the rowid indirection is pure waste.
- **Declared once in SQLAlchemy Core**, so the same schema targets Postgres by
  changing the URL: `--db postgresql+psycopg://host/polarswim`. A test compiles
  every table against the Postgres dialect to catch SQLite-only constructs.

## AI review

`polarswim review <id>` asks Claude to review one session. It is given the derived
set table and the learned parameters — and told explicitly which labels are inferred
rather than measured, so it hedges where the data is weak instead of presenting a
guess as fact.

Set `ANTHROPIC_API_KEY` in the environment or a git-ignored `.env`. Without a key the
command falls back to a deterministic offline summary, so the app is fully usable
with no credentials and the tests never make a network call.

## PySpark

`spark.py` expresses the per-set aggregation against a Spark DataFrame, installed
separately via `requirements-spark.txt`. It is **not** the default and shouldn't be:
at this scale pandas finishes in milliseconds while Spark's startup alone costs
seconds, and it needs a JVM rather than just pip. It exists so the analysis wouldn't
have to be rewritten if the length table outgrew one machine — swapping the local
frame for `spark.read.jdbc` is the only change.

## Syncing your own data

Flow has no public API for this, and its session cookie is a ~1 hour JWT with no
documented refresh — so you supply it once per sync:

1. Open **flow.polar.com** and log in
2. DevTools → **Network**, filter `api/training`
3. Open any session, right-click one of those requests → **Copy as cURL**
4. `pbpaste > ~/.polarswim/cookie.txt`

Paste the whole cURL command; `auth.py` extracts the cookie. Copy a request from
`localizations.flow.polar.com` by mistake — a separate, cookie-less host — and you
get a clear error saying so rather than a confusing 401 later. Then:

```bash
python -m polarswim sync --from 2024-01-01
```

Credentials are never written to the database or the repository.

## Limits

This reads an **undocumented internal API** that can change without notice. The
parser validates the payload shape and raises rather than silently returning empty
results, so a change surfaces as a failure instead of missing data. It requests your
own data from your own account, sequentially, with a rate floor between calls.

Stroke inference is an **estimate**, not a measurement. With no labels anywhere in
the data, its self-consistency can be measured but its accuracy cannot — and nothing
in the output pretends otherwise.

MIT licensed.
