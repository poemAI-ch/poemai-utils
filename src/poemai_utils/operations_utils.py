import datetime
import functools
import logging
import signal

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
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *_):
        self.stop_now = True
        self.stop_now_requests += 1

        if self.stop_now_requests > 2:
            _logger.info("Received 3 stop signals, stopping NOW!")
            exit(1)
