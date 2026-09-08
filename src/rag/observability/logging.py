import logging
import sys


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Logs are written to stdout so they work well locally,
    in Docker, and later in cloud environments.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
        stream=sys.stdout,
    )