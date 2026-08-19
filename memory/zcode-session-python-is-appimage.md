---
name: zcode-session-python-is-appimage
description: On this machine, any python3 exec in ZCode sessions is rerouted to
  the AppImage's Python without pip — venvs/pip installs impossible in-session
metadata:
  node_type: memory
  type: project
  originSessionId: sess_0a3c3bc7-a4da-4e36-a726-44f5d483d5aa
---

In ZCode sessions on the user's machine, every `python3` execution (even `/usr/bin/python3`) actually runs the ZCode AppImage's bundled Python 3.12 (`sys.executable` = the AppImage path). It has no pip/ensurepip, `python -m venv` produces symlinks to the AppImage with no pip inside, and get-pip bootstrap also fails.

**Why:** The agent sandbox intercepts python execution, so dependency installation and runtime testing can't be done locally in-session.

**How to apply:** Don't burn turns trying to build venvs or pip install here. Verify dependency questions via PyPI metadata (JSON API) or web fetches, syntax-check with the available python, and defer real installs/runs to the user's GPU machine. Related: [[unlimited-ocr-local-port]].
