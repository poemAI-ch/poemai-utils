from datetime import datetime, timezone


def current_time_iso():
    return datetime.now(timezone.utc).isoformat()
