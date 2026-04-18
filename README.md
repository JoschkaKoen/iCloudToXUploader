# ic2x — iCloud → X Photo Uploader

Automated pipeline that pulls photos from iCloud, runs Gemini AI safety and quality checks, optionally enhances image quality via InstructIR, queues photos for manual terminal review, then posts approved photos to X (Twitter).

## Pipeline

```mermaid
flowchart TD
    A([iCloud Photos]) -->|icloudpd --skip-videos| B[inbox/]

    B --> C{Format check\nHEIC · JPG · PNG}
    C -->|rejected| R1[rejected/format/]
    C -->|ok| D{Screenshot\ndetection}
    D -->|rejected| R2[rejected/screenshot/]
    D -->|ok| E{SHA-256\ndedup}
    E -->|seen before| R3[rejected/duplicate/]
    E -->|new| F{pHash\nperceptual dedup}
    F -->|near-duplicate| R3
    F -->|unique| G[(state.db\nstatus: seen)]

    G --> H[Gemini Safety\ngemini-2.5-flash]
    H -->|nudity · violence\nprivacy · hate…| R4[rejected/safety/]
    H -->|refused| R5[rejected/gemini_refused/]
    H -->|safe| I[Gemini Quality\ngemini-2.5-flash]
    I -->|selfie · blurry\nboring · generic| R6[rejected/quality/]
    I -->|interesting + caption| J[prepare.py\nEXIF + HEIF rotation correction\nJPEG re-encode · EXIF strip]

    J --> K{InstructIR\nenhancement\noptional}
    K -->|enabled| L[enhance.py\nsubprocess · CUDA isolated]
    K -->|disabled| M
    L --> M[queue/\nphash.jpg + phash.json]

    M --> N([ic2x review\nterminal UI])
    N -->|y — approve| O[approved/]
    N -->|n — skip| R6
    N -->|e — edit caption| O
    N -->|q — quit| EXIT([exit])

    O --> P([ic2x post])
    P -->|dry_run: true| LOG([log DRY RUN])
    P -->|dry_run: false| Q[tweepy v1.1\nmedia_upload]
    Q --> R[tweepy v2\ncreate_tweet]
    R --> S[posted/\ntweet_id in DB]

    G --> DB[(state.db)]
    S --> DB
    DB --> LOGS[logs/YYYY-MM-DD.jsonl\none JSON line per decision]
```

## Commands

```bash
ic2x run      # pull from iCloud, filter, dedup, Gemini checks, queue
ic2x review   # terminal UI: approve / skip / edit caption / quit
ic2x post     # post all approved photos to X
ic2x unstick  # reset rows stuck in 'posting' back to 'approved' for retry
```

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in API keys
# edit config.yaml: icloud.username, paths, enhance settings
```

### Required keys in `.env`

```bash
TWITTER_CONSUMER_KEY=
TWITTER_CONSUMER_SECRET=
TWITTER_ACCESS_TOKEN=
TWITTER_ACCESS_TOKEN_SECRET=
GEMINI_API_KEY=
```

### config.yaml highlights

| Key | Default | Notes |
|-----|---------|-------|
| `icloud.recent_count` | 50 | Max photos pulled per run |
| `x.dry_run` | `true` | Set `false` only when pipeline is trusted |
| `enhance.enabled` | `false` | InstructIR local enhancement (needs GPU) |
| `limits.daily_ai_calls` | 200 | Hard abort across safety + quality calls |
| `limits.hamming_threshold` | 12 | Perceptual dedup sensitivity (Hamming distance ≤ threshold) |

## Rejection stages

| Folder | Reason |
|--------|--------|
| `rejected/format/` | Non-image file type |
| `rejected/screenshot/` | Detected as device screenshot |
| `rejected/duplicate/` | SHA-256 or perceptual duplicate of seen/posted image |
| `rejected/safety/` | Gemini flagged nudity, violence, privacy, etc. |
| `rejected/gemini_refused/` | Gemini refused to analyse — treated as unsafe |
| `rejected/quality/` | Gemini judged not interesting, or reviewer said no |

## State machine

```
seen → queued → approved → posting → posted
```

`posting` is the idempotency anchor: written before the API call, `posted` written after. Any row stuck in `posting` on startup triggers a warning and halts — never silently retried.
