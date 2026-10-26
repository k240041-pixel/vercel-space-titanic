"""Production server config.  Start with:  gunicorn app.main:app -c gunicorn.conf.py

Gunicorn = process manager (spawns, monitors and restarts workers).
UvicornWorker = runs the async FastAPI app inside each worker process.
Everything is overridable with environment variables, so the same file works
on a laptop, a VPS, or a PaaS without edits.
"""
import multiprocessing
import os

# Where to listen. Most platforms (Render, Railway, Heroku, Cloud Run) inject $PORT.
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Process model. Each worker loads its own copy of the model (~tens of MB),
# so don't over-provision. Default: 2 workers per CPU core, capped at 4.
worker_class = "uvicorn_worker.UvicornWorker"
workers = int(os.getenv("WEB_CONCURRENCY", min(multiprocessing.cpu_count() * 2, 4)))

# Reliability
timeout = int(os.getenv("TIMEOUT", "60"))          # kill a worker stuck > 60s
graceful_timeout = 30                              # time to finish in-flight requests on restart
keepalive = 5
max_requests = 1000                                # recycle workers periodically to
max_requests_jitter = 100                          # guard against slow memory leaks

# Logging to stdout/stderr so the hosting platform can collect it
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
