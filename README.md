# ic2x — autonomous iCloud → X "best-of-burst" bot

A single background loop that, every ~5 hours, fetches your latest iCloud photos,
groups consecutive near-identical shots into a **burst**, asks one cheap vision
model to pick the **single best** one (and vet it for safety + quality), and posts
it to X — fully autonomously. It evaluates on thumbnails and downloads full
resolution only for the winner, so it stays cheap.

```
sync index ─▶ newest unseen burst ─▶ one VLM call picks best + vets ─▶ post
                     ▲                                                   │
                     └──────── walk back through history if nothing postable
```

## How it works

- **Photo layer** — `pyicloud` (the icloudpd binary can't list metadata or fetch
  per-asset). A local `asset_index` table is synced from iCloud **metadata only**,
  ordered by capture time. The `seen` flag on each row is the single source of
  truth, so a burst is never re-assembled and nothing is double-posted.
- **Bursts** — the newest run of consecutive, visually-similar (pHash) unseen
  stills. Videos/Live Photos are excluded by `item_type`; screenshots are dropped
  via Apple's Screenshots smart album. One multi-image VLM call returns
  `{best_index, safe, interesting, caption}`.
- **Winner** — downloaded full-res, passes a fail-closed EXIF screenshot net and
  pHash/SHA dedup, is re-encoded (EXIF stripped), then posted. Crash-safe: the
  burst is committed atomically and an interrupted post is finished on the next
  tick by `flush_pending`.
- **Walk back until postable** — if the newest burst is boring/unsafe, it falls
  through to older bursts within the same wake-up (bounded by `DAILY_AI_CALLS`).
  Lookback is unbounded.
- **Optional per-winner passes**, each fail-open and toggled in `default.env`: an
  AI orientation fix (`ROTATION_*`), same-scene dedup vs recent posts (`SCENE_DEDUP_*`),
  a GPS→city/time-grounded caption rewrite (`CAPTION_PASS_ENABLED`, `LOCATION_*`),
  and image enhancement — free local adaptive `POLISH_*` and/or Aliyun `COLOR_ENHANCE_*`.

## Setup

```bash
uv sync
cp .env.example .env      # fill ICLOUD_USERNAME/PASSWORD, TWITTER_* and one model key
ic2x login                # one-time interactive 2FA; session persists ~weeks
```

The shipped default judge model is `qwen3.7-plus` and the general fallback is
`qwen3.5-flash` — both need `DASHSCOPE_API_KEY` (Alibaba DashScope). A cheaper,
lighter alternative is `gemini-2.5-flash-lite` (needs `GEMINI_API_KEY`); the
provider is inferred from the model name, so just set `JUDGE_MODEL` accordingly.

## Commands

| Command | Purpose |
|---|---|
| `ic2x` / `ic2x bot` | Run the loop: fetch → pick best of burst → post, every `POST_INTERVAL_HOURS` |
| `ic2x login` | Interactive iCloud sign-in (2FA); also prints a live access check |
| `ic2x status` | Offline snapshot: last/next post, 24h count, lifetime totals, today's spend |
| `ic2x cost` | Show tracked AI spend per day (`--days N`) |
| `ic2x compare` | Run recent bursts through two models side by side (read-only, no posting, no DB changes) |
| `ic2x clean` | Discard non-posted image records + queue/approved files |

Calibration / diagnostics (read-only, no posting): `ic2x classify` (judge the N
newest photos), `ic2x autorotate` (test the rotation pass), `ic2x scene-dedup-test`
(tune same-scene dedup on a folder), `ic2x enhance-test` / `ic2x polish-test`
(preview image enhancement). Each writes a `*_out/<timestamp>/` folder to review.

## Burn-in → live

`X_DRY_RUN=true` (the default) advances the full state machine but posts no real
tweet. Run for a few days, read `logs/YYYY-MM-DD.jsonl` to confirm the best-of-burst
picks, safety calls, and captions look right, then set `X_DRY_RUN=false`.

```bash
ic2x compare --bursts 8                 # A/B two judge models side by side (no posting)
ic2x bot                                 # dry-run burn-in
# …looks good… set X_DRY_RUN=false in .env, restart
```

## Key settings (`default.env`)

| Variable | Default | Notes |
|---|---|---|
| `AI_DEFAULT_MODEL` | `qwen3.5-flash, 3000` | fallback model for all AI calls |
| `JUDGE_MODEL` | `qwen3.7-plus, 1000, 2000` | the best-of-burst vision judge |
| `POST_INTERVAL_HOURS` | `5` | minimum hours between posts |
| `X_DRY_RUN` | `true` | simulate posts (advances state, no tweet) |
| `BURST_HAMMING_THRESHOLD` | `8` | consecutive-shot similarity (tighter than dedup's 12) |
| `BURST_MAX_SIZE` | `5` | images per burst sent to the judge |
| `MAX_POSTS_PER_DAY` | `6` | rolling-24h safety backstop |
| `DAILY_AI_CALLS` | `200` | also bounds walk-back cost per day |
| `REAUTH_NOTIFY_CMD` | — | shell cmd run when iCloud 2FA is due |
| `ICLOUD_WITH_FAMILY` | `false` | shared library is unsafe for auto-posting |

## Deployment

Run as a long-lived service (systemd / launchd / `nohup`) on a host with a stable
network and the **iCloud cookie dir on a persistent volume** so redeploys don't
force re-auth. The trust token lasts ~weeks; when it expires the bot fires
`REAUTH_NOTIFY_CMD` and stops cleanly — re-run `ic2x login` once. Fresh-start
cutover from the old pipeline: stop it, `rm state.db`, `ic2x login`, `ic2x bot`.

State machine (`src/ic2x/status.py`): `seen → approved → posting → posted` for the
winner; losers and screenshots are marked `seen`. `posting` is the idempotency
anchor — written before the X call, `posted` after; a stuck row is auto-recovered
to `approved` at the next cycle. All decisions append to `logs/YYYY-MM-DD.jsonl`.

## Tests

All tests are offline (no iCloud / X / AI calls — clients are mocked). Run the
whole suite either way:

```bash
python tests/run_all.py                   # zero-dependency runner (no pytest needed)
uv sync --extra dev && pytest             # or pytest, if you prefer

python tests/test_burst.py                # any single file also runs standalone
python tests/run_all.py burst cost        # run_all can filter by substring
```
