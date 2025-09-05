import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional, Union

DEFAULT_INTERVAL = 8.0


class ProgressLogger:
    def __init__(
        self,
        total: Optional[int] = None,
        item_name: str = "items",
        interval: float = DEFAULT_INTERVAL,
        logger: Optional[logging.Logger] = None,
        log_func: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize a progress logger for tracking lengthy operations.

        Args:
            total: Total number of items to process (optional)
            item_name: Name for the items being processed (default: "items")
            interval: Time interval in seconds between progress logs (default: 10.0)
            logger: Logger instance to use (optional)
            log_func: Custom logging function (optional, defaults to print)
        """
        self.total = total
        self.item_name = item_name
        self.interval = interval
        self.logger = logger
        self.log_func = log_func or print

        self.processed = 0
        self.start_time = None
        self.last_log_time = None

    def __enter__(self):
        """Start the progress tracking."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        total_items_str = f", {self.total} to process" if self.total is not None else ""
        self._log(f"Started processing {self.item_name}{total_items_str}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the progress tracking and log final stats."""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            rate = self.processed / total_time if total_time > 0 else 0
            final_msg = f"Processed {self.processed}/{self.total} {self.item_name} (completed) in {total_time:.1f}s ({rate:.1f} {self.item_name}/s)"
            self._log(final_msg)

    def update(self, count: int = 1):
        """
        Update the progress counter and log if interval has passed.

        Args:
            count: Number of items processed since last update (default: 1)
        """
        self.processed = count
        current_time = time.time()
        time_since_last_log = current_time - (
            self.last_log_time if self.last_log_time else 0
        )
        # print(f"Processed: {self.processed} {self.item_name}, time since last log: {time_since_last_log:.1f}s")

        if self.last_log_time is None or time_since_last_log >= self.interval:
            self._log_progress()
            self.last_log_time = current_time

    def _log_progress(self):
        """Log the current progress."""
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0

        if self.total is not None:
            percentage = (self.processed / self.total) * 100
            msg = f"Processed {self.processed}/{self.total} {self.item_name} ({percentage:.1f}%) - {rate:.1f} {self.item_name}/s"
        else:
            msg = f"Processed {self.processed} {self.item_name} - {rate:.1f} {self.item_name}/s"

        self._log(msg)

    def _log(self, message: str):
        """Log a message using the configured logger or log function."""
        if self.logger is not None:
            self.logger.info(message)
        else:
            self.log_func(message)


@contextmanager
def progress_logger(
    total: Optional[int] = None,
    item_name: str = "items",
    interval: float = DEFAULT_INTERVAL,
    logger: Optional[logging.Logger] = None,
    log_func: Optional[Callable[[str], None]] = None,
):
    """
    Context manager for progress logging.

    Args:
        total: Total number of items to process (optional)
        item_name: Name for the items being processed (default: "items")
        interval: Time interval in seconds between progress logs (default: 10.0)
        logger: Logger instance to use (optional)
        log_func: Custom logging function (optional, defaults to print)

    Example:
        with progress_logger(total=1000, item_name="files", interval=5.0) as progress:
            for i, item in enumerate(items):
                # Do lengthy operation
                process_item(item)
                progress.update()
    """
    progress = ProgressLogger(total, item_name, interval, logger, log_func)
    with progress as p:
        yield p
