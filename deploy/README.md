# Deploying the bot

The bot runs continuously on an always-on Linux host as a **systemd user service**.
It lived on a MacBook until 2026-08-04; the Mac sleeping caused multi-hour silent gaps
and missed post slots, which is the problem this setup exists to avoid.

**Only ever run one host at a time.** The account is shared and each host keeps its own
`state.db`, so two running bots means double-posting.

## Install

```bash
cp deploy/ic2x.service ~/.config/systemd/user/ic2x.service
systemctl --user daemon-reload
systemctl --user enable --now ic2x
```

`enable` requires **lingering**, or the service dies when you log out and never starts
at boot:

```bash
loginctl show-user "$USER" | grep Linger      # want Linger=yes
sudo loginctl enable-linger "$USER"           # only if it is no
```

The unit hard-codes `/home/y/Programming/iCloudToXUploader`; edit both paths if you
deploy elsewhere.

## What a fresh clone does NOT give you

Everything the bot needs at runtime is gitignored, because the repo is public. A clone
alone will start and then fail or silently degrade. Copy these from a working host:

| path | why |
| --- | --- |
| `.env` | all credentials (iCloud, X, DashScope, Aliyun VIAPI AccessKey) |
| `state.db` | posted history, the 68k-asset catalog, and the walk-back position — **without it the bot re-posts photos** |
| `icloud_auth/` | the 2FA session; otherwise run `ic2x login` once on the host |
| `models/face/*.onnx` | YuNet + SFace for the owner-selfie gate |
| `owner_refs/` | the owner's reference photo for that gate |
| `scene_thumbs/` | thumbnails of recent posts, used to reject re-posting the same scene |

Use a consistent snapshot for the database rather than a plain copy of a live file:

```bash
python -c "import sqlite3; s=sqlite3.connect('state.db'); d=sqlite3.connect('/tmp/state.db'); s.backup(d)"
```

## Verify

```bash
.venv/bin/python tests/run_all.py     # all test files should pass
.venv/bin/ic2x status                 # read-only; no iCloud/X/AI calls
systemctl --user status ic2x
tail -f logs/xup_console.log
```

A healthy start prints the banner, `reconcile done — all in sync`, then
`🔍 scanning iCloud …`. Check the banner says `Dry run  no` — `Xup.py` forces live
posting; a bare `ic2x bot` defaults to dry-run.

Watch for two lines that mean a step is silently degraded rather than broken — both
fail open, so the bot keeps posting without them:

- `face-gate: OpenCV unavailable` → `opencv-python-headless` is missing
- `color enhance failed (…)` → the Aliyun VIAPI call did not run; the original posted

## Gotchas

- **GitHub may be unreachable from the host.** On the China-based worker `git fetch`
  dies with `gnutls_handshake() failed: The TLS connection was non-properly terminated`,
  even though iCloud, X and Aliyun are all fine. Deploy by pushing files from a machine
  that *can* reach GitHub:

  ```bash
  rsync -a src/ tests/ deploy/ pyproject.toml default.env render-worker:/home/y/Programming/iCloudToXUploader/
  rsync -a --exclude hooks/ .git/ render-worker:/home/y/Programming/iCloudToXUploader/.git/
  ```

  Do **not** `git reset` against `origin/main` on the host after a failed fetch — the
  ref is stale, so it silently moves `HEAD` backwards. (`--mixed` leaves the working
  tree alone, so the running bot is unaffected, but the repo state becomes a lie.)
- **Build the venv with one interpreter.** Running `python3.X -m venv` over an existing
  venv from a different Python leaves two `lib/pythonX.Y/site-packages` trees, with
  `pip` installing into one and the interpreter reading the other. Delete and recreate.
- **Restarts take minutes.** A stop lands mid-cycle and the pre-walk-back iCloud calls
  are not interruptible. This is expected; `TimeoutStopSec=600` covers it.
- **The post timer survives restarts** via `run_state.last_posted_at`, so restarting
  never posts immediately — and never skips a slot either.
