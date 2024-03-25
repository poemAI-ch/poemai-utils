import pytest
from poemai_utils.aws.dynamodb import VersionMismatchException
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator


def test_db(tmp_path):
    db_file = tmp_path / "test.db"
    db = DynamoDBEmulator(db_file)
    TABLE_NAME = "test_table"

    db.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", "data": "data1"})
    db.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk2", "data": "data2"})
    db.store_item(TABLE_NAME, {"pk": "pk2", "sk": "sk2", "data": "data3"})
    db.store_item(TABLE_NAME, {"pk": "pk2", "sk": "sk1", "data": "data0"})

    assert db.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk1") == {
        "pk": "pk1",
        "sk": "sk1",
        "data": "data1",
    }
    assert db.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk2") == {
        "pk": "pk1",
        "sk": "sk2",
        "data": "data2",
    }
    assert db.get_item_by_pk_sk(TABLE_NAME, "pk2", "sk2") == {
        "pk": "pk2",
        "sk": "sk2",
        "data": "data3",
    }
    assert db.get_item_by_pk_sk(TABLE_NAME, "pk2", "sk1") == {
        "pk": "pk2",
        "sk": "sk1",
        "data": "data0",
    }

    assert db.get_paginated_items_by_pk(TABLE_NAME, "pk1") == [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk1", "sk": "sk2", "data": "data2"},
    ]
    assert db.get_paginated_items_by_pk(TABLE_NAME, "pk2") == [
        {"pk": "pk2", "sk": "sk1", "data": "data0"},
        {"pk": "pk2", "sk": "sk2", "data": "data3"},
    ]

    db.delete_item_by_pk_sk(TABLE_NAME, "pk1", "sk1")
    assert db.get_paginated_items_by_pk(TABLE_NAME, "pk1") == [
        {"pk": "pk1", "sk": "sk2", "data": "data2"}
    ]
    assert db.get_paginated_items_by_pk(TABLE_NAME, "pk2") == [
        {"pk": "pk2", "sk": "sk1", "data": "data0"},
        {"pk": "pk2", "sk": "sk2", "data": "data3"},
    ]
    assert db.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk1") is None

    db.update_versioned_item_by_pk_sk(TABLE_NAME, "pk1", "sk2", {"data": "data4"}, 0)

    assert db.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk2") == {
        "pk": "pk1",
        "sk": "sk2",
        "data": "data4",
        "version": 1,
    }

    db.update_versioned_item_by_pk_sk(TABLE_NAME, "pk1", "sk2", {"data": "data5"}, 1)

    assert db.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk2") == {
        "pk": "pk1",
        "sk": "sk2",
        "data": "data5",
        "version": 2,
    }

    with pytest.raises(VersionMismatchException):
        db.update_versioned_item_by_pk_sk(
            TABLE_NAME, "pk1", "sk2", {"data": "data6"}, 1
        )

    db.store_item(TABLE_NAME, {"pk": "pk99", "sk": "sk77", "data": "data1"})
    db.store_item(TABLE_NAME, {"pk": "pk99", "sk": "sk22", "data": "data2"})
    db.store_item(TABLE_NAME, {"pk": "pk99", "sk": "sk77", "data": "data2"})

    result = db.get_paginated_items_by_pk(TABLE_NAME, "pk99")

    result_composite_keys = [(item["pk"], item["sk"]) for item in result]

    # Check that there are no duplicates
    assert len(result) == len(set(result_composite_keys))

    NUM_ITEMS = 9
    for i in range(NUM_ITEMS):
        # insert in revers
        db.store_item(
            TABLE_NAME, {"pk": "pk88", "sk": f"sk{NUM_ITEMS - i}", "data": f"data{i}"}
        )

    result = db.get_paginated_items_by_pk(TABLE_NAME, "pk88")

    result_composite_keys = [(item["pk"], item["sk"]) for item in result]

    # check that the items are sorted by the composite key (pk, sk)
    assert result_composite_keys == sorted(result_composite_keys)
