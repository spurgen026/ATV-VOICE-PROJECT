import logging
import logging.handlers

from .config import LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES, LOGS_DIR

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configuring the root logger means third-party libraries' own logging
# calls flow through it too, not just this app's - found by testing, not
# assumed: httpx and huggingface_hub log every HTTP HEAD request at INFO
# level (hf_hub_download() checking the cache), which would otherwise
# drown out the app's own INFO messages in real use. Quieted to WARNING
# specifically, not globally - this app's own loggers stay at whatever
# level configure_logging() is called with.
_NOISY_THIRD_PARTY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3")


def configure_logging(level: int = logging.INFO) -> None:
    """Sets up console + rotating file logging for the whole app. Call
    once at startup, before anything else logs - main.py does this first
    thing in run(). Replaces the print()-only approach every module used
    before this, which gave no timestamps, no severity levels, and no
    persistent record once the terminal scrolled past or closed - not
    acceptable for something meant to run unattended on a vehicle.

    File handler is rotating (see LOG_MAX_BYTES/LOG_BACKUP_COUNT in
    config.py) so a long-running process can't fill the disk."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
