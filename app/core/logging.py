"""Minimal logging setup — one place to change format later."""

import logging

from app.core.config import get_settings


def configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
