import os

DEFAULT_CONFIG = {}


def get_config_by_key(key):
    if key in os.environ:
        return os.environ[key]
    return DEFAULT_CONFIG.get(key, None)
