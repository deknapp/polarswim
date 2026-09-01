#!/bin/bash
# Double-click this to sync the latest swims and open the polarswim dashboard.
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt; }
./.venv/bin/python -m polarswim up
