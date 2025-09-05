import logging
import time
from io import StringIO
from unittest.mock import Mock, patch

import pytest
from poemai_utils.progress_logger import ProgressLogger, progress_logger


class TestProgressLogger:
    def test_basic_functionality_with_total(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with ProgressLogger(
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
        assert "10/100" in captured_logs[1] or "30/100" in captured_logs[1]
        assert "Completed processing 60 files" in captured_logs[-1]

    def test_basic_functionality_without_total(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with ProgressLogger(
            item_name="records", interval=0.1, log_func=capture_log
        ) as progress:
            progress.update(5)
            time.sleep(0.2)
            progress.update(15)

        assert len(captured_logs) >= 2
        assert "Started processing records" in captured_logs[0]
        assert "Completed processing 20 records" in captured_logs[-1]

    def test_with_logger(self):
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)

        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)

        with ProgressLogger(total=50, logger=logger, interval=0.1) as progress:
            progress.update(25)
            time.sleep(0.2)
            progress.update(25)

        log_output = log_capture.getvalue()
        assert "Started processing items" in log_output
        assert "Completed processing 50 items" in log_output

    def test_interval_timing(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with ProgressLogger(total=100, interval=0.5, log_func=capture_log) as progress:
            progress.update(10)  # Should not log yet
            progress.update(10)  # Should not log yet
            time.sleep(0.6)  # Now interval should have passed
            progress.update(10)  # Should log now

        # Should have start message, one progress update, and completion
        progress_logs = [
            log for log in captured_logs if "30/100" in log or "20/100" in log
        ]
        assert len(progress_logs) >= 1  # At least one progress log after interval

    def test_context_manager_function(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with progress_logger(
            total=200, item_name="tasks", interval=0.1, log_func=capture_log
        ) as progress:
            for i in range(0, 50, 10):
                progress.update(10)
                if i > 0:  # Skip first iteration to avoid immediate logging
                    time.sleep(0.2)

        assert len(captured_logs) >= 2
        assert "Started processing tasks" in captured_logs[0]
        assert "Completed processing 50 tasks" in captured_logs[-1]

    def test_rate_calculation(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with ProgressLogger(total=100, interval=0.1, log_func=capture_log) as progress:
            progress.update(20)
            time.sleep(0.2)
            progress.update(30)

        # Check that rate is included in messages
        completion_log = captured_logs[-1]
        assert "items/s" in completion_log or "/s" in completion_log

    def test_custom_item_name(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with ProgressLogger(
            total=10, item_name="documents", log_func=capture_log
        ) as progress:
            progress.update(5)

        start_log = captured_logs[0]
        completion_log = captured_logs[-1]

        assert "documents" in start_log
        assert "documents" in completion_log

    def test_multiple_updates_same_interval(self):
        captured_logs = []

        def capture_log(msg):
            captured_logs.append(msg)

        with ProgressLogger(total=100, interval=1.0, log_func=capture_log) as progress:
            progress.update(10)  # Should not log
            progress.update(10)  # Should not log
            progress.update(10)  # Should not log

        # Should only have start and completion messages
        assert len(captured_logs) == 2
        assert "Started processing" in captured_logs[0]
        assert "Completed processing 30" in captured_logs[-1]
