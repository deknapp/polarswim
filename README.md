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
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Put `polarswim` on your PATH. macOS ships only `python3`, without these
# dependencies, so `python -m polarswim` finds either no interpreter or no
# pandas; the launcher resolves the project's own virtualenv from anywhere.
ln -s "$PWD/bin/polarswim" ~/.local/bin/polarswim

# Everything below runs against the committed sample database — no account needed.
polarswim --db sample/sample.db status
polarswim --db sample/sample.db analyze
polarswim --db sample/sample.db card 2026-08-19     # paste into Strava
polarswim --db sample/sample.db report --from 2026-08-01
polarswim --db sample/sample.db serve                # web UI on :8770

.venv/bin/pytest -q                                 # 309 tests, no network
```

Without the symlink, every command below is `.venv/bin/python -m polarswim ...`
run from the project directory.

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
age formula or a running max would place every boundary too high.

The five bands divide **heart-rate reserve** — resting to maximum, the Karvonen
method — not zero to maximum. Dividing from zero measures against a heart rate
nobody has: it put the bottom zone below 60% of max, which is unreachable in the
water, so an easy swim reported as tempo and a steady aerobic set as threshold,
a whole zone hot. Measured from resting (79 bpm here) the bands cover the range a
swimmer actually occupies — Z1 79–135, Z5 163–172 — and Z5 becomes genuinely hard
rather than merely brisk. It is also the scheme most training plans use, which
matters for a number someone might compare against their own.

**Relative speed.** A rep's percentile against this swimmer's own reps of the **same
distance**. Ranking within an inferred stroke was tried and rejected as circular: the
classifier assigns fast lengths to freestyle, so a freestyle percentile mostly
re-expresses the classifier's threshold, and it collapsed every slow length to 0%.
Distance is measured rather than inferred, so it carries no such feedback loop.
A distance with fewer than 30 reps reports nothing instead of a fragile number.

**Personal best.** The fastest recorded time at a distance and stroke, grouped into
a tab per stroke and led by the distances that stroke is actually raced at. Two
caveats are built in: the stroke is inferred, so a best is provisional; and reps
faster than 65% of the swimmer's median pace are excluded as turn-detection artifacts
— before that filter the "best" 25 yd freestyle was 13.6 s against a 26 s median,
which is a split length, not a swim. 43 such reps are excluded, down from 45 now that
detected splits are repaired rather than only dropped.

## Uploading to Strava

Two outputs, because Strava takes two kinds of content:

**The text card** (`polarswim card <date>`) pastes into an activity description.
Plain text with coloured emoji, since that is all a description renders. Each row
carries what the dashboard table carries — effort zone, speed percentile, pace per
50 — built from the same derivation, so the two cannot disagree.

Two things it deliberately does not do. It draws **no pace bar**: the old one was
scaled to the fastest and slowest length of that single workout, so four blocks
meant nothing except "compared to the rest of today", and a shape with no scale is
worse than a number. And it **labels its fields** rather than aligning them in
columns — Strava renders descriptions in a proportional font, where padded columns
drift out of line on a phone. Zones are drawn as circles and strokes as squares,
because when both were squares a blue square meant freestyle in one column and Z2
in the next.

**The image** — the *download image for Strava* button in the dashboard — is the
full analysis as a PNG you can attach to the activity as a photo: stroke-mix donut,
per-set table with heart-rate zones, relative speed, pace per 50 and personal
bests, a labelled header for every column, and a key explaining each one. It is built server-side as SVG and rasterised in the browser through a canvas,
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
polarswim card 2026-08-19
polarswim card latest
polarswim review 2026-08-19
```

Two swims on the same day makes the date ambiguous, so it lists the candidates
instead of silently picking one.

## Web UI

```
polarswim serve
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

A set is a run of equal *distance*, though, which is not a run of one *stroke*:
four 50s freestyle then three breaststroke is one set. Rows are therefore split
wherever the stroke changes — `4×50 free` and `3×50 brst`, numbered 8a and 8b —
because a single averaged `7×50 freestyle` row is how a correction to those last
three could be saved, applied, and still appear to have done nothing.

**Repair before classify.** Polar's turn detection fails in both directions. It
misses a wall, fusing two lengths into one record; and it invents one, splitting a
single length into two impossibly fast records. Both are corrected before anything is
classified — an uncorrected merge carries a doubled time, falls past the slow end of
the distribution, and would be classified on a pace nobody swam.

A slow length is ambiguous alone — merged pair, or genuinely slow drill? — but not in
context. A merge is an **isolated near-integer multiple** of its set's median (2.0x,
3.9x); a drill set is **uniformly** slow (1.0–1.4x). Across the full dataset that
split 77 slow lengths into 38 merges and 39 real drills. A split is the mirror image
and needs a **pair** as evidence: two adjacent records, no rest between them (nobody
rests mid-length), each implausibly fast against the set median, together summing
back to one normal length. One fast record alone is just a fast length.

Both are expressed the same way — how many real pool lengths one record covers, 2–4
for a merge and 0.5 each for a split — so dividing by that factor recovers the
per-length pace in either case. The observed figure is kept alongside as
`pace_observed_s`: Polar's record is data, the correction is inference, and the two
are never conflated. 28 merges and 22 splits across 7,615 lengths.

**Three axes, no assumed ordering.** Per length we derive normalized pace (seconds
per 25 yd, so a 50 m pool is comparable), heart-rate cost above that workout's own
baseline, and the rest the set was taken on. All three matter, because per-swimmer
speed order is *not* universal — plenty of swimmers are slower at backstroke than
breaststroke. Nothing here assumes a ranking:

| | pace | cost | rest | reasoning |
|---|---|---|---|---|
| freestyle | fast | any | any | the dominant mode, and the default hypothesis |
| butterfly | mid–slow | **high** | **long** | expensive, and it buys recovery |
| breaststroke | slow | **low** | any | the glide phase makes it cheap |
| backstroke | slow | **high** | short | working hard without travelling |
| other | slow, uniform set | low | short | drill and kick share one class |

Breaststroke and a weak backstroke are indistinguishable on pace and sit in opposite
corners on cost — which is the whole reason for the second axis. Butterfly and
backstroke then sit in the *same* corner on both, and rest is what parts them: rest
is a decision the swimmer makes about how much recovery the work needs, so the set
that bought more of it cost more to swim. Rest also stops a slow set taken on long
rest being filed as drill, which is work misread as recovery.

**Set size is not a fourth axis.** A set of one or two lengths has no usable median or
spread, so every rule that reads a set statistic is reading noise there. Size does not
name a stroke; it lowers the confidence attached to the call that used it.

**It says "undetermined".** Where the evidence doesn't separate two classes, that is
the answer. On the full dataset 6% of lengths come back undetermined rather than
being assigned a coin-flip label.

**It learns.** Reference paces are estimated from the swimmer's own history and
written to `model_params`, so they tighten as workouts are synced. Keeping the model
in the database rather than a pickle makes it inspectable and diffable.

## Corrections, and the model they train

Corrections are a **sub-tab of the workout**, beside its analysis — they are a
view of one swim, and as a place of their own they meant two independent answers
to "which swim", so it was possible to correct one while looking at another.

The tab puts a stroke dropdown on every **interval**, and one on
the set that bulk-sets the intervals under it. Interval, not set, is the unit that
matters: a set is only a run of equal distances, so four 50s freestyle followed by
three breaststroke is *one set* and two strokes. Labelling per set would make that
correction unsayable and the wrong answer unfixable. The set control is there for
when it genuinely was all one thing — a 7×50 then gives fourteen labelled lengths
from one click, which is what makes hand-labelling survivable. `IM` fills in
fly · back · breast · free within an interval.

Three layers decide a length's stroke, in strict order of precedence:

| | what it is | when it speaks |
|---|---|---|
| rules | transparent thresholds over pace, cost and rest | always; the floor |
| model | Gaussian naive Bayes fitted to your corrections | 20+ labels, and only where confident |
| **correction** | what you said | **always wins** |

A correction is ground truth. It lives in its own `labels` table, outranks both
the rules and the model that trained on it, and survives re-analysis, `reparse`
and re-syncing the workout — the one thing in this database no algorithm produced.

**Why naive Bayes, in numpy, rather than scikit-learn.** Six features and six
classes fitted from tens of labels needs a low-variance model; gradient boosting
would fit the corrections better and generalise worse. Writing it out keeps the
project pip-only, but the real reason is that the fitted parameters are per-class
means and variances, which go into `model_params` as numbers you can read in SQL.
A pickled estimator would make the model the one part of this database you could
not interrogate. It also declines to run below 20 labels, and ignores any class
with fewer than 8 — a confident answer from four examples is worse than an honest
`undetermined`.

**Accuracy becomes measurable.** Without ground truth it is not measurable at all,
which is where this project started. With corrections it is: held-out accuracy and
a confusion matrix showing which strokes get confused for which. Folds hold out
whole **sets**, never lengths — lengths inside a set are near-duplicates, and
splitting on them leaks the answer across the fold boundary and flatters the score.
The number still reads pessimistically, because the sets you chose to correct are
the ones the classifier got wrong, and that is the honest direction to be wrong in.

## Medley: the one place stroke order is known

Everything above infers a stroke from how a length was swum. An individual medley is
different — fly, back, breast, free, always in that order — which makes it a
**structural** signal. Recognise the repeating four-part shape and every leg is named
without asking the classifier at all, at higher confidence than the pace rules can
offer.

Both shapes a swimmer writes down are read: `4x100 IM`, where each rep is a whole
medley, and `16x25 IM`, where the medley is broken across four reps off the wall. The
evidence is the same either way — four positions that stay consistently different
from each other across rounds, rather than wobbling around one number, with freestyle
the fastest of the four because it comes last.

Three things it refuses to claim:

- **A single four-part round.** Four lengths that happen to descend are
  indistinguishable from one medley, so at least two rounds are required and a one-off
  IM is left unlabelled rather than guessed at.
- **Any ordering between back and breast.** Which of them is slower is a fact about
  the individual swimmer, and assuming it is exactly the assumption this project
  refuses to make everywhere else.
- **A total that is not 100, 200 or 400.** Four 75s share the period-four shape and
  total 300, which is not an event anyone swims.

Medleys are ranked as their own events. A 100 IM belongs beside other 100 IMs, not
beside 100 frees, and a continuous round no longer competes for the single-stroke
best of whichever leg happened to be its mode — that had a 100 IM standing as the 100
backstroke record. A leg of a *broken* medley stays eligible, since a 25 off the wall
is a 25 either way. 16 rounds across the history — all continuous, and both
apparent broken sets turned out to be sets whose rep count was not a multiple of
four, matching only once truncated to fit.

## Personal bests, at distances that mean something

A practice throws off a fastest time at every distance a set happens to be written
at — 75s, 125s, 150s — and burying the 100 free among them makes the table
unreadable. The bests page leads with the distances each stroke is actually **raced**
at in short-course yards, one tab per stroke plus one for medley, and keeps the other
47 one click away rather than discarding them: they are still this swimmer's own
fastest times.

25 yd is included although it is not an event. It is the pool's own unit and this
swimmer's most common rep, and leaving it out would empty the table it is meant to
clarify.

## Architecture

| Module | Responsibility |
|---|---|
| `auth` | Resolves a credential — browser session first, pasted cURL as fallback; decodes the JWT's `exp` and refuses to start a long backfill on a dead session |
| `browser` | Reads the live Flow session from Chrome and renews it through Polar's own SSO redirect chain |
| `client` | Flow's private API — walks the calendar's **100-day cap**, retries 5xx, rate limits, and catches the HTML login page Flow serves with a 200 when a session lapses |
| `parse` | Pure transformation: ISO-8601 durations, per-length records, HR arrays |
| `models` | Schema as SQLAlchemy Core tables |
| `db` | Idempotent upserts, transactional loads, additive schema migration |
| `sync` | Discover, skip stored, fetch, parse, load |
| `analyze` | Sets, turn-defect repair, features, classification, medley detection, learned parameters |
| `learn` | Fits a stroke model to the swimmer's corrections, and reports held-out accuracy |
| `render` | Unicode cards |
| `report` | pandas aggregation over a date range |
| `ai` | Optional Claude review of one session |
| `web` | Local Flask UI |
| `spark` | Optional PySpark path (see below) |
| `bin/polarswim` | Launcher that finds the project's virtualenv from any directory |

## Schema

```
workouts ──┬── lengths       (workout_id, idx)   one row per pool length
           ├── hr_samples    (workout_id, t_s)   flattened from values[] + interval
           ├── raw_payloads  (workout_id)        untouched API response
           ├── predictions   (workout_id, idx)   inference, kept apart from observation
           └── labels        (workout_id, idx)   corrections; outrank both
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
- **A new column reaches an existing database.** `create_all` creates missing tables
  but never alters one that already exists, so adding a column would otherwise leave
  every already-synced database a column short — a failure that surfaces much later
  as an error on a column the code is certain exists. Connecting runs an additive
  migration that adds missing nullable columns, and deliberately refuses to guess at
  anything more invasive: a dropped column, a changed type, a new primary key raise
  `SchemaDrift` instead. The fix for those is `polarswim reparse`, which rebuilds
  every derived table from the stored raw payloads with no network and no credential.

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

```bash
polarswim sync
```

If you are signed in to Flow in Chrome, that is the whole procedure. It picks up
everything since your last sync; `--from 2024-01-01` backfills further. Sync borrows
the session the browser already has, and runs the analysis when it finishes.

**Why that takes any work at all.** Flow has no public API here, and it hands its
web client a `FLOW_SESSION` JWT that expires after exactly one hour with no
refresh claim — so a credential copied by hand is dead by the next morning. But
the hourly token is not how the browser stays logged in. Behind it sits a
`remember-me` cookie on `auth.polar.com` with a **two-week** life, which the
browser silently trades for a new token on every page load. That is why the site
never asks you for a password while a copied token dies overnight.

So `auth.py` takes the credential that *mints* tokens rather than a minted one:
it reads the `*.polar.com` cookies from the local Chrome profile — decrypting
them with the key Chrome keeps in the macOS Keychain, which prompts for approval
once — and if the token is spent it replays the same `/flowSso/login` redirect
chain the browser performs. The login *page* is a JavaScript application, but the
silent re-authentication behind it is ordinary redirects and cookies, so this
needs no headless browser and adds no dependency. You re-authenticate about once
a fortnight, by opening Flow in Chrome, and only if that login has lapsed.

Two details that decide whether this is safe to do. `remember-me` is **not**
rotated by minting, so the tool cannot log your browser out. `session_id` **is**,
so whoever minted last holds the live session — the renewed cookies are therefore
saved to `~/.polarswim/session.json` (mode `600`) rather than re-read from a
Chrome copy the tool's own last run invalidated, and a stale store falls through
to Chrome to recover.

Only hosts under `polar.com` are ever decrypted. The Keychain key opens every
cookie Chrome holds, and a tool that reads swims has no business touching a bank
session; a test enforces the boundary.

**The manual path still works**, and `auto` falls back to it whenever the browser
cannot help — an unsupported platform, a declined Keychain prompt, a lapsed
login. Use `--cookie-source file` to insist on it:

1. Open **flow.polar.com** and log in
2. DevTools → **Network**, filter `api/training`
3. Open any session, right-click one of those requests → **Copy as cURL**
4. `pbpaste > ~/.polarswim/cookie.txt`

Paste the whole cURL command; `auth.py` extracts the cookie. Copy a request from
`localizations.flow.polar.com` by mistake — a separate, cookie-less host — and you
get a clear error saying so rather than a confusing 401 later.

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
