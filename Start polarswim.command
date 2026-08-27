#!/bin/bash
# Double-click this to launch the polarswim dashboard.
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt; }
./.venv/bin/python -m polarswim serve &
sleep 2 && open http://127.0.0.1:8770
wait
