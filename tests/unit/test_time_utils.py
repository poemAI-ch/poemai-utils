from datetime import datetime, timezone

from poemai_utils.time_utils import parse_time_iso, semantic_date_difference


def test_parse_time_iso():
    assert parse_time_iso("2023-03-15T12:00:00Z") == datetime(
        2023, 3, 15, 12, 0, 0, tzinfo=timezone.utc
    )
    assert parse_time_iso("2023-03-15T12:00:00+02:00") == datetime(
        2023, 3, 15, 10, 0, 0, tzinfo=timezone.utc
    )
    assert parse_time_iso("2023-03-15T12:00:00") == datetime(
        2023, 3, 15, 12, 0, 0, tzinfo=timezone.utc
    )
    assert parse_time_iso("2023-03-15T12:00:00.123456Z") == datetime(
        2023, 3, 15, 12, 0, 0, 123456, tzinfo=timezone.utc
    )
    assert parse_time_iso("2023-03-15T07:00:00-05:00") == datetime(
        2023, 3, 15, 12, 0, 0, tzinfo=timezone.utc
    )
    assert parse_time_iso("2023-01-01T00:00:00Z") == datetime(
        2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc
    )


def test_semantic_date_difference():

    date_1 = datetime(2023, 3, 15, tzinfo=timezone.utc).date()
    date_2 = datetime(2023, 3, 20, tzinfo=timezone.utc).date()
    assert semantic_date_difference(date_1, date_2) == "in 5 Tagen"
