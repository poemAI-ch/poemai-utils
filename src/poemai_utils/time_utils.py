from datetime import date, datetime, timezone

from dateutil.relativedelta import relativedelta


def current_time_iso():
    return datetime.now(timezone.utc).isoformat()


def semantic_date_difference(
    reference_date: date, other_date: date, language="de"
) -> str:

    if language != "de":
        raise NotImplementedError("Only German language is supported")

    diff = relativedelta(other_date, reference_date)

    # Determine past or future
    if other_date > reference_date:
        prefix = "in"
        past = False
    elif other_date < reference_date:
        prefix = "vor"
        past = True
    else:
        return "Heute"

    # Handle different time units
    if abs(diff.years) >= 1:
        if abs(diff.years) == 1:
            return f"{prefix} einem Jahr" if not past else f"{prefix} etwa einem Jahr"
        else:
            return f"{prefix} {abs(diff.years)} Jahren"

    if abs(diff.months) >= 1:
        if abs(diff.months) == 1:
            return f"{prefix} einem Monat"
        else:
            return f"{prefix} {abs(diff.months)} Monaten"

    if abs(diff.days) >= 7:
        weeks = abs(diff.days) // 7
        if weeks == 1:
            return f"{prefix} einer Woche"
        else:
            return f"{prefix} {weeks} Wochen"

    if abs(diff.days) > 0:
        if abs(diff.days) == 1:
            return f"{prefix} einem Tag"
        else:
            return f"{prefix} {abs(diff.days)} Tagen"

    return "Heute"


def parse_time_iso(s: str) -> datetime:
    """
    Parses an ISO 8601 timestamp string into a timezone-aware UTC datetime object.
    Handles both 'Z' (UTC) and offset timezones.
    """
    if s.endswith("Z"):
        # Replace Z with +00:00 for fromisoformat compatibility
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # Treat naive datetime as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Convert any tz-aware datetime to UTC
        dt = dt.astimezone(timezone.utc)
    return dt
