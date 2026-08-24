from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging once."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

