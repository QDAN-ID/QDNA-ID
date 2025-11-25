# config.py
"""
Project Title: QDNA-ID: Quantum Device Native Authentication
Developed by: Osamah N. Neamah
Department of Mechatronic Engineering, Graduate Institute, Karabuk University, Karabuk, Turkey

Scheduler (hardware-only):
- Discover real devices and run sequentially with stagger
"""

import os, logging, time
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timezone
import devices
from challenge import run_session_on_hardware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

RUN_INTERVAL_HOURS = int(os.environ.get("QDNA_RUN_INTERVAL_HOURS", "12"))
STAGGER_SECONDS    = int(os.environ.get("QDNA_STAGGER_SECONDS", "30"))
DEFAULT_SHOTS      = int(os.environ.get("QDNA_SHOTS", "1024"))
MIN_SHOTS, MAX_SHOTS = 256, 1024

def _norm(s: int) -> int:
    return MIN_SHOTS if s < MIN_SHOTS else MAX_SHOTS if s > MAX_SHOTS else s

def job_dispatch():
    logger.info("Dispatcher @ %s", datetime.now(timezone.utc).isoformat())
    devs = devices.list_real_devices()
    if not devs:
        logger.error("No real IBM devices available. Aborting this cycle.")
        return
    shots = _norm(DEFAULT_SHOTS)
    for i, d in enumerate(devs):
        try:
            logger.info("Running on %s (shots=%d)", d["name"], shots)
            run_session_on_hardware(backend=d["name"], shots=shots)
        except Exception as e:
            logger.exception("Error on %s: %s", d["name"], e)
        if i < len(devs)-1:
            time.sleep(STAGGER_SECONDS)

def start_scheduler():
    sch = BackgroundScheduler(timezone="UTC")
    sch.add_job(job_dispatch, 'interval', hours=RUN_INTERVAL_HOURS, next_run_time=datetime.now(timezone.utc))
    sch.start()
    logger.info("Scheduler started (every %sh)", RUN_INTERVAL_HOURS)
    try:
        while True: time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sch.shutdown()

if __name__ == "__main__":
    start_scheduler()
