"""configure_logging() tests. Production hardening: replaced print()
throughout the app with real logging (levels, timestamps, a persistent
rotating file) - a print()-only app gives no record once the terminal
scrolls past or closes, unacceptable for something meant to run
unattended on a vehicle.
"""

import logging

from resort_atv_voice.config import LOG_FILE
from resort_atv_voice.logging_config import configure_logging


def _reset_root_logger():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_configure_logging_creates_the_log_directory():
    _reset_root_logger()
    try:
        configure_logging()
        assert LOG_FILE.parent.is_dir()
    finally:
        _reset_root_logger()


def test_configure_logging_adds_a_console_and_a_file_handler():
    _reset_root_logger()
    try:
        configure_logging()
        root = logging.getLogger()
        handler_types = {type(h).__name__ for h in root.handlers}
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types
    finally:
        _reset_root_logger()


def test_configure_logging_sets_the_requested_level():
    _reset_root_logger()
    try:
        configure_logging(level=logging.WARNING)
        assert logging.getLogger().level == logging.WARNING
    finally:
        _reset_root_logger()


def test_noisy_third_party_loggers_are_quieted_to_warning():
    # httpx/huggingface_hub log every HTTP HEAD request at INFO level
    # (confirmed by testing, not assumed) - would drown out the app's
    # own messages once the root logger is configured to capture
    # everything, not just this app's loggers.
    _reset_root_logger()
    try:
        configure_logging(level=logging.INFO)
        for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        _reset_root_logger()


def test_a_logged_message_actually_reaches_the_file():
    _reset_root_logger()
    try:
        configure_logging()
        marker = "test_a_logged_message_actually_reaches_the_file marker"
        logging.getLogger("resort_atv_voice.test").info(marker)
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert marker in LOG_FILE.read_text(encoding="utf-8")
    finally:
        _reset_root_logger()
