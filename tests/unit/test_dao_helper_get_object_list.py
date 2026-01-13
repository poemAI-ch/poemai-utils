from enum import Enum

from poemai_utils.aws.dao_helper import DaoHelper
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator


class FakeKey(str, Enum):
    OBJECT_TYPE = "OBJECT_TYPE"
    HASH = "HASH"
    RANGE = "RANGE"


class FakeObjectType(str, Enum):
    ITEM = "ITEM"


# Attach the attributes DaoHelper expects
FakeObjectType.ITEM.pk_components = [FakeKey.HASH]
FakeObjectType.ITEM.sk_components = [FakeKey.RANGE]
FakeObjectType.ITEM.required_fields = [
    FakeKey.HASH,
    FakeKey.RANGE,
]
FakeObjectType.ITEM.to_drop_fields = []


def test_get_object_list_returns_prefix_matches_in_order():
    db = DynamoDBEmulator(
        None, allowed_reserved_keywords=["hash", "range", "object_type"]
    )
    table = "tbl"
    fmt = {}

    # Store three items with same pk, varying sk
    for rng in ["001", "002", "010"]:
        DaoHelper.store_object(
            FakeKey,
            fmt,
            db,
            table,
            FakeObjectType.ITEM,
            {"hash": "h1", "range": rng},
        )

    # Only items starting with RANGE prefix "" should be returned, in order
    items = list(
        DaoHelper.get_object_list(
            FakeKey, fmt, db, table, FakeObjectType.ITEM, {"hash": "h1"}, "range"
        )
    )
    assert [i["range"] for i in items] == ["001", "002", "010"]


def test_get_object_list_stops_after_prefix_changes():
    db = DynamoDBEmulator(
        None, allowed_reserved_keywords=["hash", "range", "object_type"]
    )
    table = "tbl"
    fmt = {}

    # Items with same pk but different range prefixes
    DaoHelper.store_object(
        FakeKey,
        fmt,
        db,
        table,
        FakeObjectType.ITEM,
        {"hash": "h1", "range": "100"},
    )
    DaoHelper.store_object(
        FakeKey,
        fmt,
        db,
        table,
        FakeObjectType.ITEM,
        {"hash": "h1", "range": "200"},
    )

    # Query starting at range prefix "" should only return items starting with computed sk
    items = list(
        DaoHelper.get_object_list(
            FakeKey, fmt, db, table, FakeObjectType.ITEM, {"hash": "h1"}, "range"
        )
    )
    assert [i["range"] for i in items] == ["100", "200"]
