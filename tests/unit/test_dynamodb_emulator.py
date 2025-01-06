import logging
import uuid
from collections import defaultdict

import pytest
from poemai_utils.aws.dynamodb import DynamoDB, VersionMismatchException
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator

_logger = logging.getLogger(__name__)


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

    assert db.get_paginated_items_by_pk(
        TABLE_NAME, "pk2", projection_expression="pk,sk"
    ) == [
        {"pk": "pk2", "sk": "sk1"},
        {"pk": "pk2", "sk": "sk2"},
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


def test_get_paginated_items_starting_at_pk_sk(tmp_path):

    db_file = tmp_path / "test.db"
    ddb = DynamoDBEmulator(db_file)
    TEST_TABLE_NAME = "test_table"

    num_pks = 10
    num_sks = 10

    for i in range(num_pks):
        for j in range(num_sks):
            ddb.store_item(
                TEST_TABLE_NAME,
                {"pk": f"pk{i}", "sk": f"sk{j}", "data": f"data{i}{j}"},
            )

    pk = "pk3"

    start_sk = "sk4"

    paginated_items = ddb.get_paginated_items_starting_at_pk_sk(
        TEST_TABLE_NAME, pk, start_sk
    )

    paginated_items_list = list(paginated_items)
    for item in paginated_items_list:
        _logger.info(f"item: {item}")

    assert len(paginated_items_list) == 6

    assert all([item["pk"] == pk for item in paginated_items_list])
    assert all([item["sk"] >= start_sk for item in paginated_items_list])


def test_paginated_items_starting_at_pk_sk_sorting(tmp_path):

    db_file = tmp_path / "test.db"
    ddb = DynamoDBEmulator(db_file)
    TEST_TABLE_NAME = "test_table"

    nun_pks = 5
    num_sks = 5

    pks_sks = []

    for i in range(nun_pks):
        pk = uuid.uuid4().hex
        for j in range(num_sks):
            sk = uuid.uuid4().hex

            pks_sks.append((pk, sk))

    for i, (pk, sk) in enumerate(pks_sks):
        ddb.store_item(
            TEST_TABLE_NAME,
            {"pk": pk, "sk": sk, "data": f"Item{i}"},
        )

    sorted_pks_sks = sorted(pks_sks)

    by_pk = defaultdict(list)

    for pk, sk in sorted_pks_sks:
        by_pk[pk].append((pk, sk))

    for pk, keys_list in by_pk.items():

        for i, start_sk in enumerate([kli[1] for kli in keys_list]):

            paginated_items = ddb.get_paginated_items_starting_at_pk_sk(
                TEST_TABLE_NAME, pk, start_sk
            )

            paginated_items_list = list(paginated_items)

            assert len(paginated_items_list) == len(keys_list) - i

            for j, item in enumerate(paginated_items_list):
                assert item["pk"] == pk
                assert item["sk"] == keys_list[i + j][1]


def test_batch_get_items_by_pk_sk():

    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    ddb.store_item(TEST_TABLE_NAME, {"pk": "pk1", "sk": "sk1", "data": "data1"})

    ddb.store_item(TEST_TABLE_NAME, {"pk": "pk1", "sk": "sk2", "data": "data2"})
    ddb.store_item(TEST_TABLE_NAME, {"pk": "pk2", "sk": "sk2", "data": "data3"})

    test_keys = [
        {"pk": "pk1", "sk": "sk1"},
        {"pk": "pk1", "sk": "sk2"},
        {"pk": "pk2", "sk": "sk2"},
    ]
    test_keys = [DynamoDB.dict_to_item(item) for item in test_keys]

    items = ddb.batch_get_items_by_pk_sk(TEST_TABLE_NAME, test_keys)

    assert len(items) == 3

    for item in items:
        assert item["pk"] in ["pk1", "pk2"]
        assert item["sk"] in ["sk1", "sk2"]
        assert item["data"] in ["data1", "data2", "data3"]
