import logging
import time
from io import StringIO
from unittest.mock import Mock, patch

import pytest
from poemai_utils.progress_logger import ProgressLogger, progress_logger


def test_basic_functionality_with_total():
    captured_logs = []

    def capture_log(msg):
        captured_logs.append(msg)

    with progress_logger(
        total=100, item_name="files", interval=0.1, log_func=capture_log
    ) as progress:
        progress.update(10)
        time.sleep(0.2)  # Ensure interval passes
        progress.update(20)
        time.sleep(0.2)
        progress.update(30)

    # Should have start, 2 progress updates, and completion messages
    assert len(captured_logs) >= 3
    assert "Started processing files" in captured_logs[0]
