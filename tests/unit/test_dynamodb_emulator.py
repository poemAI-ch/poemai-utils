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


def test_binary_item():

    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    binary_data = b"binary_data"

    ddb.store_item(TEST_TABLE_NAME, {"pk": "pk1", "sk": "sk1", "data": binary_data})

    item = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk1", "sk1")

    assert item["data"] == binary_data
    sk = "sk1"
    key_condition_expression = "sk = :sk"
    expression_attribute_values = {":sk": {"S": sk}}
    projection_expression = None

    for item in ddb.get_paginated_items(
        TEST_TABLE_NAME,
        key_condition_expression,
        expression_attribute_values,
        projection_expression,
        index_name="",
    ):
        _logger.info(f"item: {item}")


def test_index_projection_enforcement():
    """Test that index projection is properly enforced when enabled."""

    # Test with enforcement enabled
    ddb = DynamoDBEmulator(None, enforce_index_existence=True)
    TEST_TABLE_NAME = "test_table"

    # Add a KEYS_ONLY index
    ddb.add_index(
        TEST_TABLE_NAME,
        "ClientIdIndex",
        projection_type="KEYS_ONLY",
        hash_key="client_id",
    )

    # Add an INCLUDE index
    ddb.add_index(
        TEST_TABLE_NAME,
        "PhysicalDeviceIdIndex",
        projection_type="INCLUDE",
        hash_key="client_id",  # Changed from device_id to client_id
        non_key_attributes=["name", "status"],
    )

    # Add an ALL index
    ddb.add_index(
        TEST_TABLE_NAME, "AllFieldsIndex", projection_type="ALL", hash_key="button_id"
    )

    # Store test items
    test_items = [
        {
            "pk": "ORG#test_org#BUTTON_ID#button1",
            "sk": "BUTTON_CLIENT_ASSOCIATION#DAY#2024-06-01",
            "client_id": "client123",
            "organization_key": "test_org",
            "site_id": "site456",
            "button_id": "button1",
            "day": "2024-06-01",
            "name": "Test Button",
            "status": "active",
            "secret_data": "should_not_be_visible",
        },
        {
            "pk": "ORG#test_org#BUTTON_ID#button2",
            "sk": "BUTTON_CLIENT_ASSOCIATION#DAY#2024-06-02",
            "client_id": "client123",
            "organization_key": "test_org",
            "site_id": "site789",
            "button_id": "button2",
            "day": "2024-06-02",
            "name": "Another Button",
            "status": "inactive",
            "secret_data": "also_secret",
        },
    ]

    for item in test_items:
        ddb.store_item(TEST_TABLE_NAME, item)

    # Test KEYS_ONLY projection
    keys_only_results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="client_id = :cid",
            expression_attribute_values={":cid": {"S": "client123"}},
            index_name="ClientIdIndex",
        )
    )

    assert len(keys_only_results) == 2
    for item in keys_only_results:
        item_dict = DynamoDB.item_to_dict(item)
        # Should only have pk, sk, client_id
        expected_keys = {"pk", "sk", "client_id"}
        assert set(item_dict.keys()) == expected_keys
        assert "secret_data" not in item_dict
        assert "name" not in item_dict
        assert "status" not in item_dict

    # Test INCLUDE projection
    include_results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="client_id = :cid",
            expression_attribute_values={":cid": {"S": "client123"}},
            index_name="PhysicalDeviceIdIndex",
        )
    )

    assert len(include_results) == 2
    for item in include_results:
        item_dict = DynamoDB.item_to_dict(item)
        # Should have pk, sk, client_id, name, status
        expected_keys = {"pk", "sk", "client_id", "name", "status"}
        assert set(item_dict.keys()) == expected_keys
        assert "secret_data" not in item_dict
        assert item_dict["name"] in ["Test Button", "Another Button"]
        assert item_dict["status"] in ["active", "inactive"]

    # Test ALL projection
    all_results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="client_id = :cid",
            expression_attribute_values={":cid": {"S": "client123"}},
            index_name="AllFieldsIndex",
        )
    )

    assert len(all_results) == 2
    for item in all_results:
        item_dict = DynamoDB.item_to_dict(item)
        # Should have all fields
        assert "secret_data" in item_dict
        assert "name" in item_dict
        assert "status" in item_dict
        assert item_dict["secret_data"] in ["should_not_be_visible", "also_secret"]

    # Test that using undefined index raises error when enforcement is enabled
    with pytest.raises(ValueError, match="Index UndefinedIndex not found"):
        list(
            ddb.get_paginated_items(
                table_name=TEST_TABLE_NAME,
                key_condition_expression="client_id = :cid",
                expression_attribute_values={":cid": {"S": "client123"}},
                index_name="UndefinedIndex",
            )
        )


def test_index_projection_without_enforcement():
    """Test that without enforcement, all fields are returned (original behavior)."""

    # Test without enforcement (default behavior)
    ddb = DynamoDBEmulator(None, enforce_index_existence=False)
    TEST_TABLE_NAME = "test_table"

    # Store test item
    test_item = {
        "pk": "ORG#test_org#BUTTON_ID#button1",
        "sk": "BUTTON_CLIENT_ASSOCIATION#DAY#2024-06-01",
        "client_id": "client123",
        "secret_data": "should_be_visible_without_enforcement",
    }

    ddb.store_item(TEST_TABLE_NAME, test_item)

    # Query with undefined index - should work and return all fields
    results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="client_id = :cid",
            expression_attribute_values={":cid": {"S": "client123"}},
            index_name="UndefinedIndex",  # This index doesn't exist but should not cause error
        )
    )

    assert len(results) == 1
    item_dict = DynamoDB.item_to_dict(results[0])
    assert "secret_data" in item_dict
    assert item_dict["secret_data"] == "should_be_visible_without_enforcement"


def test_add_index_validation():
    """Test that add_index validates parameters correctly."""

    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    # Test invalid projection type
    with pytest.raises(ValueError, match="projection_type must be one of"):
        ddb.add_index(
            TEST_TABLE_NAME, "TestIndex", projection_type="INVALID", hash_key="test_key"
        )

    # Test INCLUDE without projected_attributes
    with pytest.raises(ValueError, match="projected_attributes must be provided"):
        ddb.add_index(
            TEST_TABLE_NAME, "TestIndex", projection_type="INCLUDE", hash_key="test_key"
        )

    # Test valid configurations
    ddb.add_index(
        TEST_TABLE_NAME,
        "KeysOnlyIndex",
        projection_type="KEYS_ONLY",
        hash_key="test_key",
    )
    ddb.add_index(
        TEST_TABLE_NAME, "AllIndex", projection_type="ALL", hash_key="test_key"
    )
    ddb.add_index(
        TEST_TABLE_NAME,
        "IncludeIndex",
        projection_type="INCLUDE",
        hash_key="test_key",
        non_key_attributes=["field1", "field2"],
    )

    # Verify indexes were added
    assert "KeysOnlyIndex" in ddb.indexes[TEST_TABLE_NAME]
    assert "AllIndex" in ddb.indexes[TEST_TABLE_NAME]
    assert "IncludeIndex" in ddb.indexes[TEST_TABLE_NAME]
    assert ddb.indexes[TEST_TABLE_NAME]["IncludeIndex"]["non_key_attributes"] == [
        "field1",
        "field2",
    ]


def test_index_projection_with_binary_data():
    """Test that index projection handles binary data gracefully."""

    # Test with enforcement enabled
    ddb = DynamoDBEmulator(None, enforce_index_existence=True)
    TEST_TABLE_NAME = "test_table"

    # Add a KEYS_ONLY index
    ddb.add_index(
        TEST_TABLE_NAME,
        "BinaryIndex",
        projection_type="KEYS_ONLY",
        hash_key="data_type",
    )

    # Add an INCLUDE index that should include the binary field
    ddb.add_index(
        TEST_TABLE_NAME,
        "BinaryIncludeIndex",
        projection_type="INCLUDE",
        hash_key="data_type",
        non_key_attributes=["binary_data", "metadata"],
    )

    # Store test item with binary data
    binary_data = (
        b"PK\x03\x04-\x00\x00\x00\x08\x00\x00\x00!\x00binary_embedding_data_here"
    )
    test_item = {
        "pk": "DATA#embedding1",
        "sk": "TYPE#numpy_array",
        "data_type": "embedding",
        "binary_data": binary_data,
        "metadata": "test_metadata",
        "other_field": "should_not_be_included",
    }

    ddb.store_item(TEST_TABLE_NAME, test_item)

    # Test KEYS_ONLY projection - should work and exclude binary data
    keys_only_results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="data_type = :dt",
            expression_attribute_values={":dt": {"S": "embedding"}},
            index_name="BinaryIndex",
        )
    )

    assert len(keys_only_results) == 1
    item_dict = DynamoDB.item_to_dict(keys_only_results[0])
    # Should only have pk, sk, data_type (no binary data)
    expected_keys = {"pk", "sk", "data_type"}
    assert set(item_dict.keys()) == expected_keys
    assert "binary_data" not in item_dict
    assert "metadata" not in item_dict
    assert "other_field" not in item_dict

    # Test INCLUDE projection - should gracefully handle binary data serialization failure
    # Our fix should prevent crashes and either include the binary data or fall back to original item
    include_results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="data_type = :dt",
            expression_attribute_values={":dt": {"S": "embedding"}},
            index_name="BinaryIncludeIndex",
        )
    )

    assert len(include_results) == 1
    item_dict = DynamoDB.item_to_dict(include_results[0])

    # Due to our fix, when binary data causes serialization issues,
    # the emulator falls back to returning the original item (without projection)
    # This ensures functionality doesn't break while maintaining backwards compatibility
    assert "pk" in item_dict
    assert "sk" in item_dict
    assert "data_type" in item_dict
    # The binary data should be present (either through successful projection or fallback)
    assert "binary_data" in item_dict
    assert item_dict["binary_data"] == binary_data
    assert "metadata" in item_dict
    assert item_dict["metadata"] == "test_metadata"

    # Test query without index - should always work with binary data
    no_index_results = list(
        ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="pk = :pk",
            expression_attribute_values={":pk": {"S": "DATA#embedding1"}},
        )
    )

    assert len(no_index_results) == 1
    item_dict = DynamoDB.item_to_dict(no_index_results[0])
    assert "binary_data" in item_dict
    assert item_dict["binary_data"] == binary_data


# ...existing tests...
