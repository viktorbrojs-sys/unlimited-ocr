---
name: github-cli-setup
description: gh CLI authenticated as viktorbrojs-sys (SSH protocol), git
  identity configured — agent can commit/push/create repos on request
metadata:
  node_type: memory
  type: user
  originSessionId: sess_0a3c3bc7-a4da-4e36-a726-44f5d483d5aa
---

GitHub environment on the user's local PC (verified 2026-08-16): git 2.43 + gh CLI 2.45 installed; `gh auth status` → logged in as **viktorbrojs-sys** (keyring token, `repo` scope, SSH protocol for git ops). Global git identity: `viktor` / `viktor.bro.js@gmail.com`. The github.com SSH host key was added to `~/.ssh/known_hosts` (the very first push failed on host-key verification otherwise).

**Why:** enables the agreed workflow — the user says «закоммить и запушь» and the agent runs add/commit/push with zero extra setup.

**How to apply:** For new repos ask the user about visibility first (for unlimited-ocr they chose **private** — repo lives at https://github.com/viktorbrojs-sys/unlimited-ocr, branch `main`). Push over the existing SSH remote. The GPU PC (srs-mint-work) has GitHub configured **via PAT over HTTPS, no gh CLI** (user stated 2026-08-17); link recipe there: `git init -b main` → `git remote add origin https://github.com/viktorbrojs-sys/unlimited-ocr.git` → `git fetch` → `git reset --hard origin/main`. Caveat (2026-08-17): that PC's stored PAT does **not** see the private repo — `git fetch` fails «Repository not found» without prompting (helper auto-supplies credentials). Working fallback: put the username in the remote URL (`https://viktorbrojs-sys@github.com/...`) and fetch once with `git -c credential.helper=` to force a PAT prompt; a 403 on push would mean the PAT lacks `repo` scope, a 404 means wrong account/no repo access — re-issue the token if needed. Related: [[unlimited-ocr-local-port]].
