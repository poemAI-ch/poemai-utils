import datetime
import functools
import logging
import os
import signal
import threading

_logger = logging.getLogger(__name__)


def log_call_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        current_time = datetime.datetime.now()
        _logger.info(f"Call: {func.__name__}.")
        result = func(*args, **kwargs)
        delta = datetime.datetime.now() - current_time

        _logger.info(f"Call: {func.__name__} took {delta.total_seconds()}s.")
        return result

    return wrapper


class GracefulStopper:
    stop_now = False
    stop_now_requests = 0

    def __init__(self):
        self.signal_handling_enabled = False
        self._setup_signal_handling()

    def _setup_signal_handling(self):
        """
        Set up signal handling with situational awareness.
        Signal handling only works in the main thread of the main interpreter.
        """
        try:
            # Check if we're in the main thread
            is_main_thread = threading.current_thread() is threading.main_thread()

            # Check for AWS Lambda environment
            is_lambda = bool(
                os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
                or os.environ.get("LAMBDA_TASK_ROOT")
                or os.environ.get("_HANDLER")
            )

            # Check for other serverless/container environments that might not support signals
            is_serverless = is_lambda or bool(os.environ.get("AWS_EXECUTION_ENV"))

            if not is_main_thread:
                _logger.debug(
                    "GracefulStopper: Not in main thread, signal handling disabled"
                )
                return

            if is_serverless:
                _logger.debug(
                    "GracefulStopper: Serverless environment detected, signal handling disabled"
                )
                return

            # Try to set up signal handlers
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
            self.signal_handling_enabled = True
            _logger.debug("GracefulStopper: Signal handling enabled")

        except (ValueError, OSError) as e:
            # ValueError: signal only works in main thread of the main interpreter
            # OSError: can occur in some restricted environments
            _logger.debug(f"GracefulStopper: Signal handling not available: {e}")
            self.signal_handling_enabled = False

    def exit_gracefully(self, *_):
        """Handle graceful shutdown signals."""
        self.stop_now = True
        self.stop_now_requests += 1

        _logger.info(
            f"GracefulStopper: Received stop signal (request #{self.stop_now_requests})"
        )

        if self.stop_now_requests > 2:
            _logger.info("GracefulStopper: Received 3 stop signals, stopping NOW!")
            exit(1)
