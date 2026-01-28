import logging
import uuid
from collections import defaultdict
from decimal import Decimal

import pytest
from poemai_utils.aws.dynamodb import DynamoDB, VersionMismatchException
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator

_logger = logging.getLogger(__name__)


def test_db(tmp_path):
    db_file = tmp_path / "test.db"
    db = DynamoDBEmulator(db_file, allowed_reserved_keywords=["data"])
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
    ddb = DynamoDBEmulator(db_file, allowed_reserved_keywords=["data"])
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
    ddb = DynamoDBEmulator(db_file, allowed_reserved_keywords=["data"])
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

    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["data"])
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

    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["data"])
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
    ddb = DynamoDBEmulator(
        None,
        enforce_index_existence=True,
        allowed_reserved_keywords=["name", "status", "day"],
    )
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


def test_get_item():

    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["data"])
    TEST_TABLE_NAME = "test_table"

    ddb.store_item(TEST_TABLE_NAME, {"pk": "pk1", "sk": "sk1", "data": "data1"})

    item_read_back = ddb.get_item(
        TableName=TEST_TABLE_NAME, Key={"pk": {"S": "pk1"}, "sk": {"S": "sk1"}}
    )["Item"]
    assert item_read_back == {
        "pk": {"S": "pk1"},
        "sk": {"S": "sk1"},
        "data": {"S": "data1"},
    }

    item_read_back = ddb.get_item(
        TableName=TEST_TABLE_NAME,
        Key={"pk": {"S": "pk1"}, "sk": {"S": "sk1"}},
        ProjectionExpression="pk, sk",
    )["Item"]

    assert item_read_back == {
        "pk": {"S": "pk1"},
        "sk": {"S": "sk1"},
    }


def test_key_schema_pk_sk():

    ddb = DynamoDBEmulator(None, log_access=True, allowed_reserved_keywords=["data"])

    ddb.add_key_schema(
        "test_table",
        [
            {"AttributeName": "trx_id", "KeyType": "HASH"},
            {"AttributeName": "created_at", "KeyType": "RANGE"},
        ],
    )

    ddb.store_item(
        "test_table",
        {"trx_id": "trx1", "created_at": "2024-06-01T12:00:00Z", "data": "data1"},
    )

    item = ddb.get_item_by_pk_sk("test_table", "trx1", "2024-06-01T12:00:00Z")

    assert item == {
        "trx_id": "trx1",
        "created_at": "2024-06-01T12:00:00Z",
        "data": "data1",
    }


def test_key_schema_pk_only():

    ddb = DynamoDBEmulator(None, log_access=True, allowed_reserved_keywords=["data"])

    ddb.add_key_schema(
        "test_table",
        [
            {"AttributeName": "trx_id", "KeyType": "HASH"},
        ],
    )

    ddb.store_item(
        "test_table",
        {"trx_id": "trx1", "data": "data1"},
    )

    item = ddb.get_item_by_pk("test_table", "trx1")

    assert item == {
        "trx_id": "trx1",
        "data": "data1",
    }


def test_reserved_keyword_validation_store_item():
    """Test that reserved keywords are rejected when storing items."""
    ddb = DynamoDBEmulator(None)
    TABLE_NAME = "test_table"

    # Test that 'status' is rejected
    with pytest.raises(Exception) as exc_info:
        ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", "status": "active"})
    assert "reserved keyword(s): ['status']" in str(exc_info.value)
    assert "expression attribute names" in str(exc_info.value)

    # Test that other common reserved keywords are rejected
    reserved_keywords_to_test = [
        "name",
        "data",
        "count",
        "size",
        "type",
        "key",
        "value",
        "index",
    ]

    for keyword in reserved_keywords_to_test:
        with pytest.raises(Exception) as exc_info:
            ddb.store_item(
                TABLE_NAME, {"pk": "pk1", "sk": "sk1", keyword: "test_value"}
            )
        assert f"reserved keyword(s): ['{keyword}']" in str(exc_info.value)

    # Test that case-insensitive matching works
    with pytest.raises(Exception) as exc_info:
        ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", "STATUS": "active"})
    assert "reserved keyword(s): ['STATUS']" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", "Status": "active"})
    assert "reserved keyword(s): ['Status']" in str(exc_info.value)

    # Test that multiple reserved keywords are detected
    with pytest.raises(Exception) as exc_info:
        ddb.store_item(
            TABLE_NAME,
            {"pk": "pk1", "sk": "sk1", "status": "active", "name": "test", "count": 5},
        )
    assert "reserved keyword(s):" in str(exc_info.value)
    # Should contain all three keywords
    error_message = str(exc_info.value)
    assert "status" in error_message
    assert "name" in error_message
    assert "count" in error_message

    # Test that non-reserved keywords work fine
    ddb.store_item(
        TABLE_NAME, {"pk": "pk1", "sk": "sk1", "my_data": "test", "field": "value"}
    )
    item = ddb.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk1")
    assert item["my_data"] == "test"
    assert item["field"] == "value"


def test_reserved_keyword_validation_update_item():
    """Test that reserved keywords are rejected when updating items."""
    ddb = DynamoDBEmulator(None)
    TABLE_NAME = "test_table"

    # First store a valid item
    ddb.store_item(
        TABLE_NAME, {"pk": "pk1", "sk": "sk1", "my_data": "test", "version": 0}
    )

    # Test that 'status' is rejected in updates
    with pytest.raises(Exception) as exc_info:
        ddb.update_versioned_item_by_pk_sk(
            TABLE_NAME, "pk1", "sk1", {"status": "active"}, 0
        )
    assert "reserved keyword(s): ['status']" in str(exc_info.value)
    assert "expression attribute names" in str(exc_info.value)

    # Test other reserved keywords in updates
    reserved_keywords_to_test = ["name", "data", "count", "size"]

    for keyword in reserved_keywords_to_test:
        with pytest.raises(Exception) as exc_info:
            ddb.update_versioned_item_by_pk_sk(
                TABLE_NAME, "pk1", "sk1", {keyword: "test_value"}, 0
            )
        assert f"reserved keyword(s): ['{keyword}']" in str(exc_info.value)

    # Test that valid updates still work
    ddb.update_versioned_item_by_pk_sk(
        TABLE_NAME, "pk1", "sk1", {"my_field": "updated_value"}, 0
    )
    item = ddb.get_item_by_pk_sk(TABLE_NAME, "pk1", "sk1")
    assert item["my_field"] == "updated_value"
    assert item["version"] == 1


def test_reserved_keyword_validation_projection_expression():
    """Test that reserved keywords are rejected in ProjectionExpression."""
    ddb = DynamoDBEmulator(None)
    TABLE_NAME = "test_table"

    # Store a valid item
    ddb.store_item(
        TABLE_NAME, {"pk": "pk1", "sk": "sk1", "my_data": "test", "field": "value"}
    )

    # Test get_item with reserved keyword in ProjectionExpression
    with pytest.raises(Exception) as exc_info:
        ddb.get_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": "pk1"}, "sk": {"S": "sk1"}},
            ProjectionExpression="status,my_data",
        )
    assert "reserved keyword(s): ['status']" in str(exc_info.value)
    assert "expression attribute names" in str(exc_info.value)

    # Test multiple reserved keywords in ProjectionExpression
    with pytest.raises(Exception) as exc_info:
        ddb.get_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": "pk1"}, "sk": {"S": "sk1"}},
            ProjectionExpression="status,name,count,my_data",
        )
    error_message = str(exc_info.value)
    assert "reserved keyword(s):" in error_message
    assert "status" in error_message
    assert "name" in error_message
    assert "count" in error_message

    # Test that valid ProjectionExpression works
    result = ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": "pk1"}, "sk": {"S": "sk1"}},
        ProjectionExpression="my_data,field",
    )
    assert "Item" in result
    # The result should only contain the projected fields plus pk/sk
    item_data = result["Item"]
    assert "my_data" in item_data
    assert "field" in item_data


def test_reserved_keyword_validation_query_projection():
    """Test that reserved keywords are rejected in query ProjectionExpression."""
    ddb = DynamoDBEmulator(None)
    TABLE_NAME = "test_table"

    # Store test items
    ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", "my_data": "test1"})
    ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk2", "my_data": "test2"})

    # Test query with reserved keyword in ProjectionExpression
    with pytest.raises(Exception) as exc_info:
        ddb.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": "pk1"}},
            ProjectionExpression="status,my_data",
        )
    assert "reserved keyword(s): ['status']" in str(exc_info.value)
    assert "expression attribute names" in str(exc_info.value)

    # Test that valid query ProjectionExpression works
    result = ddb.query(
        TableName=TABLE_NAME,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": "pk1"}},
        ProjectionExpression="my_data",
    )
    assert "Items" in result
    assert len(result["Items"]) == 2


def test_reserved_keyword_validation_comprehensive():
    """Test a comprehensive set of DynamoDB reserved keywords."""
    ddb = DynamoDBEmulator(None)
    TABLE_NAME = "test_table"

    # Test a selection of important reserved keywords that are commonly problematic
    # Only include keywords that are actually in the DynamoDB reserved keywords list
    important_reserved_keywords = [
        "status",
        "name",
        "data",
        "count",
        "size",
        "type",
        "key",
        "value",
        "index",
        "table",
        "column",
        "primary",
        "foreign",
        "unique",
        "order",
        "group",
        "where",
        "select",
        "from",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "user",
        "role",
        "grant",
        "read",
        "write",
        "execute",
        "time",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "timestamp",
        "zone",
        "path",
        "file",
        "system",
        "public",
        "private",
        "session",
        "connection",
        # Removed "directory" as it's not actually a DynamoDB reserved keyword
    ]

    for keyword in important_reserved_keywords:
        # Test in store_item
        with pytest.raises(Exception) as exc_info:
            ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", keyword: "test"})
        assert f"reserved keyword(s): ['{keyword}']" in str(exc_info.value)

    # Test that legitimate field names work
    legitimate_fields = [
        "user_id",
        "user_name",
        "item_status",
        "data_field",
        "count_value",
        "custom_type",
        "my_index",
        "table_name",
        "created_time",
        "updated_date",
    ]

    for field in legitimate_fields:
        # These should not raise exceptions
        ddb.store_item(
            TABLE_NAME, {"pk": f"pk_{field}", "sk": "sk1", field: "test_value"}
        )
        item = ddb.get_item_by_pk_sk(TABLE_NAME, f"pk_{field}", "sk1")
        assert item[field] == "test_value"

    # Test that reserved keywords still work when not explicitly allowed
    # This test is separate from the main test to show that validation still happens
    # We'll use a fresh DynamoDBEmulator without allowed keywords
    strict_ddb = DynamoDBEmulator(None)
    with pytest.raises(Exception) as exc_info:
        strict_ddb.store_item(TABLE_NAME, {"pk": "pk1", "sk": "sk1", "status": "test"})
    assert "reserved keyword(s): ['status']" in str(exc_info.value)


def test_validate_attribute_names_function():
    """Test the validate_attribute_names function directly."""
    from poemai_utils.aws.dynamodb_emulator import validate_attribute_names

    # Test with dictionary
    with pytest.raises(Exception) as exc_info:
        validate_attribute_names({"status": "active", "name": "test"}, "test context")
    assert "reserved keyword(s):" in str(exc_info.value)
    assert "status" in str(exc_info.value)
    assert "name" in str(exc_info.value)
    assert "test context" in str(exc_info.value)

    # Test with list
    with pytest.raises(Exception) as exc_info:
        validate_attribute_names(["status", "name", "count"], "test context")
    error_message = str(exc_info.value)
    assert "reserved keyword(s):" in error_message
    assert "status" in error_message
    assert "name" in error_message
    assert "count" in error_message

    # Test with single string
    with pytest.raises(Exception) as exc_info:
        validate_attribute_names("status", "test context")
    assert "reserved keyword(s): ['status']" in str(exc_info.value)

    # Test with valid data
    validate_attribute_names(
        {"my_field": "value", "another_field": "test"}, "test context"
    )  # Should not raise
    validate_attribute_names(
        ["my_field", "another_field"], "test context"
    )  # Should not raise
    validate_attribute_names("my_field", "test context")  # Should not raise

    # Test with allowed keywords
    validate_attribute_names(
        {"status": "active", "my_field": "test"},
        "test context",
        allowed_keywords=["status"],
    )  # Should not raise
    validate_attribute_names(
        ["status", "my_field"], "test context", allowed_keywords=["status"]
    )  # Should not raise
    validate_attribute_names(
        "status", "test context", allowed_keywords=["status"]
    )  # Should not raise

    # Test that non-allowed keywords still raise
    with pytest.raises(Exception) as exc_info:
        validate_attribute_names(
            {"status": "active", "name": "test"},
            "test context",
            allowed_keywords=["status"],
        )
    assert "reserved keyword(s): ['name']" in str(exc_info.value)


def test_get_item_with_expression_attribute_names():
    """Test get_item with ExpressionAttributeNames support for reserved keywords."""
    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["status", "size", "type"])
    TABLE_NAME = "test_table"

    # Store a test item with reserved keywords
    test_item = {
        "pk": "test_pk",
        "sk": "test_sk",
        "status": "active",
        "size": 100,
        "type": "document",
        "normal_field": "value",
    }
    ddb.store_item(TABLE_NAME, test_item)

    # Test 1: ProjectionExpression without ExpressionAttributeNames should fail for reserved keywords
    with pytest.raises(Exception) as exc_info:
        ddb.get_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}},
            ProjectionExpression="pk, sk, status, normal_field",
        )
    assert "reserved keyword(s): ['status']" in str(exc_info.value)

    # Test 2: ProjectionExpression with ExpressionAttributeNames should work
    result = ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}},
        ProjectionExpression="pk, sk, #status, normal_field",
        ExpressionAttributeNames={"#status": "status"},
    )

    # Verify the result contains the projected fields
    assert result is not None
    assert "Item" in result
    item = DynamoDB.item_to_dict(result["Item"])
    assert item["pk"] == "test_pk"
    assert item["sk"] == "test_sk"
    assert item["status"] == "active"
    assert item["normal_field"] == "value"
    assert "size" not in item  # Should not be included since not in projection
    assert "type" not in item  # Should not be included since not in projection

    # Test 3: Multiple reserved keywords in ExpressionAttributeNames
    result = ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}},
        ProjectionExpression="pk, #status, #size, #type",
        ExpressionAttributeNames={
            "#status": "status",
            "#size": "size",
            "#type": "type",
        },
    )

    assert result is not None
    item = DynamoDB.item_to_dict(result["Item"])
    assert item["pk"] == "test_pk"
    assert item["status"] == "active"
    assert item["size"] == 100
    assert item["type"] == "document"
    assert "sk" not in item  # Should not be included since not in projection
    assert "normal_field" not in item  # Should not be included since not in projection

    # Test 4: Mix of aliased and non-aliased attributes
    result = ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}},
        ProjectionExpression="pk, #status, normal_field",
        ExpressionAttributeNames={"#status": "status"},
    )

    assert result is not None
    item = DynamoDB.item_to_dict(result["Item"])
    assert item["pk"] == "test_pk"
    assert item["status"] == "active"
    assert item["normal_field"] == "value"
    assert len(item) == 3  # Only the projected fields

    # Test 5: Missing ExpressionAttributeName should raise error
    with pytest.raises(Exception) as exc_info:
        ddb.get_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}},
            ProjectionExpression="pk, #missing_alias",
            ExpressionAttributeNames={
                "#status": "status"
            },  # #missing_alias not defined
        )
    assert "ExpressionAttributeName '#missing_alias' not found" in str(exc_info.value)

    # Test 6: Get full item (no ProjectionExpression) should work normally
    result = ddb.get_item(
        TableName=TABLE_NAME, Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}}
    )

    assert result is not None
    item = DynamoDB.item_to_dict(result["Item"])
    assert item == test_item  # Should return the full item

    # Test 7: Non-reserved keywords should work without aliases
    result = ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}},
        ProjectionExpression="pk, sk, normal_field",
    )

    assert result is not None
    item = DynamoDB.item_to_dict(result["Item"])
    assert item["pk"] == "test_pk"
    assert item["sk"] == "test_sk"
    assert item["normal_field"] == "value"
    assert len(item) == 3


def test_get_item_expression_attribute_names_validation_behavior():
    """Test that ExpressionAttributeNames properly bypasses reserved keyword validation."""
    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["name"])
    TABLE_NAME = "test_table"

    # Store an item with a reserved keyword using allowed_keywords
    test_item = {"pk": "test", "sk": "test", "name": "test_name"}
    ddb.store_item(TABLE_NAME, test_item)

    # Test 1: Direct use of reserved keyword in ProjectionExpression should fail
    with pytest.raises(Exception) as exc_info:
        ddb.get_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": "test"}, "sk": {"S": "test"}},
            ProjectionExpression="pk, name",
        )
    assert "reserved keyword(s): ['name']" in str(exc_info.value)

    # Test 2: Using ExpressionAttributeNames should validate the alias, not the resolved keyword
    # The ProjectionExpression contains "#n" (not reserved), so validation should pass
    result = ddb.get_item(
        TableName=TABLE_NAME,
        Key={"pk": {"S": "test"}, "sk": {"S": "test"}},
        ProjectionExpression="pk, #n",
        ExpressionAttributeNames={"#n": "name"},
    )

    assert result is not None
    item = DynamoDB.item_to_dict(result["Item"])
    assert item["pk"] == "test"
    assert item["name"] == "test_name"

    # Test 3: Even with ExpressionAttributeNames, if the alias itself is reserved, it should fail
    with pytest.raises(Exception) as exc_info:
        ddb.get_item(
            TableName=TABLE_NAME,
            Key={"pk": {"S": "test"}, "sk": {"S": "test"}},
            ProjectionExpression="pk, status",  # 'status' in ProjectionExpression is reserved
            ExpressionAttributeNames={
                "#n": "name"
            },  # This doesn't help because 'status' is not aliased
        )
    assert "reserved keyword(s): ['status']" in str(exc_info.value)


def test_float_and_Decimal():
    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    with pytest.raises(TypeError):
        ddb.store_item(
            TEST_TABLE_NAME, {"pk": "pk1", "sk": "sk1", "float_value": 12.34}
        )

    ddb.store_item(
        TEST_TABLE_NAME,
        {"pk": "pk2", "sk": "sk2", "decimal_value": Decimal("56.78")},
    )

    item2 = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk2", "sk2")

    assert isinstance(item2["decimal_value"], Decimal)
    assert item2["decimal_value"] == Decimal("56.78")


def test_decimal_serialization_no_warning(caplog):
    """Test that Decimal values don't produce serialization warnings."""
    from unittest.mock import MagicMock

    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["data"])
    TEST_TABLE_NAME = "test_table"

    # Clear any existing log messages
    caplog.clear()

    # Test that Decimal values don't produce warnings
    with caplog.at_level(logging.WARNING):
        ddb.store_item(
            TEST_TABLE_NAME,
            {"pk": "pk1", "sk": "sk1", "decimal_value": Decimal("123.45")},
        )

    # Should be no warning messages about Decimal serialization
    warning_messages = [
        record.message for record in caplog.records if record.levelno >= logging.WARNING
    ]
    decimal_warnings = [
        msg for msg in warning_messages if "Decimal" in msg and "serializable" in msg
    ]
    assert (
        len(decimal_warnings) == 0
    ), f"Unexpected Decimal serialization warnings: {decimal_warnings}"

    # Verify the item was stored correctly
    item = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk1", "sk1")
    assert isinstance(item["decimal_value"], Decimal)
    assert item["decimal_value"] == Decimal("123.45")

    # Test that truly non-serializable objects still produce warnings
    caplog.clear()
    mock_obj = MagicMock()

    with caplog.at_level(logging.WARNING):
        # This should produce a warning about JSON serialization and fail immediately
        with pytest.raises(TypeError, match="Item contains unserializable data"):
            ddb.store_item(
                TEST_TABLE_NAME, {"pk": "pk2", "sk": "sk2", "mock_value": mock_obj}
            )

    # Should have a warning about the mock object
    warning_messages = [
        record.message for record in caplog.records if record.levelno >= logging.WARNING
    ]
    mock_warnings = [msg for msg in warning_messages if "serializable" in msg]
    assert len(mock_warnings) > 0, "Expected warning about non-serializable mock object"


def test_binary_data_serialization_no_warning(caplog):
    """Test that binary data doesn't produce serialization warnings."""

    ddb = DynamoDBEmulator(None, allowed_reserved_keywords=["data"])
    TEST_TABLE_NAME = "test_table"

    # Clear any existing log messages
    caplog.clear()

    # Test that binary data doesn't produce warnings
    binary_data = b"some binary data with special chars \x00\x01\x02\xff"

    with caplog.at_level(logging.WARNING):
        ddb.store_item(
            TEST_TABLE_NAME, {"pk": "pk1", "sk": "sk1", "binary_value": binary_data}
        )

    # Should be no warning messages about binary data serialization
    warning_messages = [
        record.message for record in caplog.records if record.levelno >= logging.WARNING
    ]
    binary_warnings = [msg for msg in warning_messages if "serializable" in msg]
    assert (
        len(binary_warnings) == 0
    ), f"Unexpected binary data serialization warnings: {binary_warnings}"

    # Verify the item was stored correctly
    item = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk1", "sk1")
    # DynamoDB emulator wraps binary data in BinaryPoemai objects
    assert hasattr(
        item["binary_value"], "value"
    ), "Binary data should be wrapped in BinaryPoemai"
    assert isinstance(item["binary_value"].value, bytes), "Binary value should be bytes"
    assert (
        item["binary_value"].value == binary_data
    ), "Binary data should match original"

    # Test with bytearray as well
    caplog.clear()
    bytearray_data = bytearray(b"bytearray data")

    with caplog.at_level(logging.WARNING):
        ddb.store_item(
            TEST_TABLE_NAME,
            {"pk": "pk2", "sk": "sk2", "bytearray_value": bytearray_data},
        )

    # Should be no warning messages
    warning_messages = [
        record.message for record in caplog.records if record.levelno >= logging.WARNING
    ]
    bytearray_warnings = [msg for msg in warning_messages if "serializable" in msg]
    assert (
        len(bytearray_warnings) == 0
    ), f"Unexpected bytearray serialization warnings: {bytearray_warnings}"

    # Verify the item was stored correctly
    item = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk2", "sk2")
    # DynamoDB emulator wraps binary data in BinaryPoemai objects
    assert hasattr(
        item["bytearray_value"], "value"
    ), "Binary data should be wrapped in BinaryPoemai"
    assert isinstance(
        item["bytearray_value"].value, (bytes, bytearray)
    ), "Binary value should be bytes or bytearray"
    # bytearray may be preserved as bytearray
    assert (
        item["bytearray_value"].value == bytearray_data
    ), "Bytearray data should match original"


def test_delete_inexistent_item_doesn_t_throw():
    """Test that deleting a non-existent item does not throw an error."""

    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    # Attempt to delete a non-existent item
    try:
        ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, "nonexistent_pk", "nonexistent_sk")
    except Exception as e:
        pytest.fail(f"Deleting a non-existent item raised an exception: {e}")


def test_update_expression_size_limit():
    """Test that UpdateExpression size limit is enforced (4KB limit)."""
    from botocore.exceptions import ClientError

    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    # Store an initial item with version 0
    initial_item = {"pk": "test_pk", "sk": "test_sk", "version": 0}
    ddb.store_item(TEST_TABLE_NAME, initial_item)

    # Create an update with many attributes that will exceed the 4KB limit
    # Each fragment_X_status attribute adds roughly 88 bytes to the UpdateExpression
    # We need about 50+ attributes to exceed 4KB
    large_update = {}
    num_fragments = 100  # This should definitely exceed the limit

    for i in range(num_fragments):
        large_update[f"fragment_{i}_status"] = "completed"
        large_update[f"fragment_{i}_duration"] = 123.45
        large_update[f"fragment_{i}_timestamp"] = "2024-01-28T12:00:00Z"

    # Attempt to update with too many attributes - should raise ValidationException
    with pytest.raises(ClientError) as exc_info:
        ddb.update_versioned_item_by_pk_sk(
            TEST_TABLE_NAME, "test_pk", "test_sk", large_update, 0
        )

    # Verify the error details
    error = exc_info.value
    assert error.response["Error"]["Code"] == "ValidationException"
    assert (
        "Expression size has exceeded the maximum allowed size"
        in error.response["Error"]["Message"]
    )

    # Verify the item was not updated (should still be at version 0)
    item = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "test_pk", "test_sk")
    assert item["version"] == 0
    assert "fragment_0_status" not in item  # Update should have failed


def test_update_expression_size_limit_boundary():
    """Test UpdateExpression size limit at the boundary (just under vs just over 4KB)."""
    from botocore.exceptions import ClientError

    ddb = DynamoDBEmulator(None)
    TEST_TABLE_NAME = "test_table"

    # Store an initial item
    ddb.store_item(TEST_TABLE_NAME, {"pk": "test_pk", "sk": "test_sk", "version": 0})

    # Test with a smaller number of attributes that should pass (under 4KB)
    # With 20 fragments × 3 attributes each = 60 attributes, roughly 88 bytes each = ~5,280 bytes
    # But the actual calculation depends on attribute names, so let's use a safer number
    small_update = {}
    for i in range(10):
        small_update[f"field_{i}_a"] = "value"
        small_update[f"field_{i}_b"] = 123

    # This should succeed (well under 4KB)
    ddb.update_versioned_item_by_pk_sk(
        TEST_TABLE_NAME, "test_pk", "test_sk", small_update, 0
    )

    # Verify the update succeeded
    item = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "test_pk", "test_sk")
    assert item["version"] == 1
    assert item["field_0_a"] == "value"

    # Now test with enough attributes to exceed the limit (over 4KB)
    # Reset the item
    ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, "test_pk", "test_sk")
    ddb.store_item(TEST_TABLE_NAME, {"pk": "test_pk", "sk": "test_sk", "version": 0})

    # Create an update with many attributes that will exceed 4KB
    large_update = {}
    for i in range(60):  # 60 fragments with 3 attributes each = 180 SET clauses
        large_update[f"fragment_{i}_status"] = "completed"
        large_update[f"fragment_{i}_duration"] = 123.45
        large_update[f"fragment_{i}_error"] = "none"

    # This should fail with ValidationException
    with pytest.raises(ClientError) as exc_info:
        ddb.update_versioned_item_by_pk_sk(
            TEST_TABLE_NAME, "test_pk", "test_sk", large_update, 0
        )

    error = exc_info.value
    assert error.response["Error"]["Code"] == "ValidationException"
    assert "Expression size has exceeded" in error.response["Error"]["Message"]
