#!/usr/bin/env bash
# Start the API the way it will run in production.
set -euo pipefail
cd "$(dirname "$0")"
exec gunicorn app.main:app -c gunicorn.conf.py
