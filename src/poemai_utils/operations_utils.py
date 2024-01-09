import datetime
import functools
import logging

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
