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
