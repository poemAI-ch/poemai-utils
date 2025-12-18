import logging
import uuid
from collections import defaultdict
from types import SimpleNamespace

import pytest
from poemai_utils.aws.dynamodb import DynamoDB

_logger = logging.getLogger(__name__)

TEST_TABLE_NAME = "poemai-integration-test"

TEST_ITEM = {
    "pk": "TEST#1",
    "sk": "TEST_SK#1",
    "content": "some content",
}


# this function will be run before all tests in this file
def setup_module():
    # store the test item, if not present
    ddb = create_dynamo_db()
    if ddb.get_item_by_pk_sk(TEST_TABLE_NAME, TEST_ITEM["pk"], TEST_ITEM["sk"]) is None:
        ddb.store_item(TEST_TABLE_NAME, TEST_ITEM)


# this function will be run after all tests in this file
def teardown_module():
    ddb = create_dynamo_db()
    # delete the test item, if present
    if (
        ddb.get_item_by_pk_sk(TEST_TABLE_NAME, TEST_ITEM["pk"], TEST_ITEM["sk"])
        is not None
    ):
        ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, TEST_ITEM["pk"], TEST_ITEM["sk"])

    # delete all items with pk in primary key
    items_to_delete = ddb.scan_for_items_by_pk_sk(
        TEST_TABLE_NAME, pk_contains="pk", sk_contains=None
    )
    for item in items_to_delete:
        ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, item["pk"], item["sk"])


def test_store_item(ddb: DynamoDB):
    item_to_store = {"pk": "pk1", "sk": "sk1", "data": "data1"}
    ddb.store_item(TEST_TABLE_NAME, item_to_store)

    item_read_back = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk1", "sk1")
    assert item_read_back == item_to_store


def test_load_item(ddb: DynamoDB):
    item_read_back = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "TEST#1", "TEST_SK#1")
    assert item_read_back == {
        "pk": "TEST#1",
        "sk": "TEST_SK#1",
        "content": "some content",
    }


def test_delete_item(ddb: DynamoDB):
    item_to_store = {"pk": "pk1", "sk": "sk1", "data": "data1"}
    ddb.store_item(TEST_TABLE_NAME, item_to_store)

    item_to_delete = {"pk": {"S": "pk1"}, "sk": {"S": "sk1"}}
    ddb.delete_item(TEST_TABLE_NAME, item_to_delete)

    item_read_back = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk1", "sk1")
    assert item_read_back is None


def test_delete_item_by_pk_sk(ddb: DynamoDB):
    item_to_store = {"pk": "pk1", "sk": "sk1", "data": "data1"}
    ddb.store_item(TEST_TABLE_NAME, item_to_store)

    item_to_delete = {"pk": {"S": "pk1"}, "sk": {"S": "sk1"}}
    ddb.delete_item_by_pk_sk(
        TEST_TABLE_NAME, item_to_delete["pk"]["S"], item_to_delete["sk"]["S"]
    )

    item_read_back = ddb.get_item_by_pk_sk(TEST_TABLE_NAME, "pk1", "sk1")
    assert item_read_back is None


def test_get_item(ddb: DynamoDB):
    item_read_back = ddb.get_item(
        TableName=TEST_TABLE_NAME, Key={"pk": {"S": "TEST#1"}, "sk": {"S": "TEST_SK#1"}}
    )["Item"]
    assert item_read_back == {
        "pk": {"S": "TEST#1"},
        "sk": {"S": "TEST_SK#1"},
        "content": {"S": "some content"},
    }

    item_read_back = ddb.get_item(
        TableName=TEST_TABLE_NAME,
        Key={"pk": {"S": "TEST#1"}, "sk": {"S": "TEST_SK#1"}},
        ProjectionExpression="pk, content",
    )["Item"]

    assert item_read_back == {
        "pk": {"S": "TEST#1"},
        "content": {"S": "some content"},
    }


def test_get_item_with_expression_attribute_names(ddb: DynamoDB):
    """Test ExpressionAttributeNames support in get_item method."""
    # Store an item with fields that could be reserved keywords
    test_item = {
        "pk": "attr_test",
        "sk": "test_sk",
        "name": "test_name",
        "status": "active",
        "normal_field": "value",
    }
    ddb.store_item(TEST_TABLE_NAME, test_item)

    try:
        # Test 1: Use ExpressionAttributeNames to access reserved keywords
        result = ddb.get_item(
            TableName=TEST_TABLE_NAME,
            Key={"pk": {"S": "attr_test"}, "sk": {"S": "test_sk"}},
            ProjectionExpression="pk, #n, #s",
            ExpressionAttributeNames={"#n": "name", "#s": "status"},
        )

        # Verify the result contains the projected fields
        assert "Item" in result
        item_dict = DynamoDB.item_to_dict(result["Item"])

        # Should only contain the projected fields
        expected_keys = {"pk", "name", "status"}
        actual_keys = set(item_dict.keys())
        assert (
            actual_keys == expected_keys
        ), f"Expected keys {expected_keys}, got {actual_keys}"

        # Verify the values
        assert item_dict["pk"] == "attr_test"
        assert item_dict["name"] == "test_name"
        assert item_dict["status"] == "active"
        assert (
            "normal_field" not in item_dict
        )  # Should not be included since not in projection

        print("✅ Test 1 passed: ExpressionAttributeNames works with get_item")

        # Test 2: Mix of aliased and non-aliased attributes
        result = ddb.get_item(
            TableName=TEST_TABLE_NAME,
            Key={"pk": {"S": "attr_test"}, "sk": {"S": "test_sk"}},
            ProjectionExpression="pk, #n, normal_field",
            ExpressionAttributeNames={"#n": "name"},
        )

        assert "Item" in result
        item_dict = DynamoDB.item_to_dict(result["Item"])

        expected_keys = {"pk", "name", "normal_field"}
        actual_keys = set(item_dict.keys())
        assert actual_keys == expected_keys

        assert item_dict["pk"] == "attr_test"
        assert item_dict["name"] == "test_name"
        assert item_dict["normal_field"] == "value"
        assert "status" not in item_dict

        print("✅ Test 2 passed: Mixed aliased and non-aliased attributes work")

        # Test 3: Get full item (no ProjectionExpression) should work normally
        result = ddb.get_item(
            TableName=TEST_TABLE_NAME,
            Key={"pk": {"S": "attr_test"}, "sk": {"S": "test_sk"}},
        )

        assert "Item" in result
        item_dict = DynamoDB.item_to_dict(result["Item"])
        assert item_dict == test_item  # Should return the full item

        print(
            "✅ Test 3 passed: Full item retrieval without ProjectionExpression works"
        )

    finally:
        # Clean up the test item
        ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, "attr_test", "test_sk")


def test_query_with_expression_attribute_names(ddb: DynamoDB):
    """Test ExpressionAttributeNames support in query method."""
    # Store test items with fields that could be reserved keywords
    test_items = [
        {
            "pk": "query_test",
            "sk": "item1",
            "name": "first_item",
            "status": "active",
            "count": 1,
        },
        {
            "pk": "query_test",
            "sk": "item2",
            "name": "second_item",
            "status": "inactive",
            "count": 2,
        },
    ]

    for item in test_items:
        ddb.store_item(TEST_TABLE_NAME, item)

    try:
        # Test query with ExpressionAttributeNames
        result = ddb.query(
            TableName=TEST_TABLE_NAME,
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": "query_test"}},
            ProjectionExpression="pk, sk, #n, #s",
            ExpressionAttributeNames={"#n": "name", "#s": "status"},
        )

        # Verify the result
        assert "Items" in result
        assert len(result["Items"]) == 2

        for item in result["Items"]:
            item_dict = DynamoDB.item_to_dict(item)
            expected_keys = {"pk", "sk", "name", "status"}
            actual_keys = set(item_dict.keys())
            assert (
                actual_keys == expected_keys
            ), f"Expected keys {expected_keys}, got {actual_keys}"

            assert item_dict["pk"] == "query_test"
            assert item_dict["sk"] in ["item1", "item2"]
            assert item_dict["name"] in ["first_item", "second_item"]
            assert item_dict["status"] in ["active", "inactive"]
            assert (
                "count" not in item_dict
            )  # Should not be included since not in projection

        print("✅ Query with ExpressionAttributeNames test passed")

    finally:
        # Clean up the test items
        for item in test_items:
            ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, item["pk"], item["sk"])


def test_get_paginated_items_with_expression_attribute_names(ddb: DynamoDB):
    """Test ExpressionAttributeNames support in get_paginated_items method."""
    # Store test items with fields that could be reserved keywords
    test_items = [
        {
            "pk": "paginated_test",
            "sk": "item1",
            "name": "first_item",
            "status": "active",
        },
        {
            "pk": "paginated_test",
            "sk": "item2",
            "name": "second_item",
            "status": "inactive",
        },
    ]

    for item in test_items:
        ddb.store_item(TEST_TABLE_NAME, item)

    try:
        # Test get_paginated_items with ExpressionAttributeNames
        paginated_items = ddb.get_paginated_items(
            table_name=TEST_TABLE_NAME,
            key_condition_expression="pk = :pk",
            expression_attribute_values={":pk": {"S": "paginated_test"}},
            projection_expression="pk, sk, #n, #s",
            expression_attribute_names={"#n": "name", "#s": "status"},
        )

        items_list = list(paginated_items)
        assert len(items_list) == 2

        for item in items_list:
            item_dict = DynamoDB.item_to_dict(item)
            expected_keys = {"pk", "sk", "name", "status"}
            actual_keys = set(item_dict.keys())
            assert (
                actual_keys == expected_keys
            ), f"Expected keys {expected_keys}, got {actual_keys}"

            assert item_dict["pk"] == "paginated_test"
            assert item_dict["sk"] in ["item1", "item2"]
            assert item_dict["name"] in ["first_item", "second_item"]
            assert item_dict["status"] in ["active", "inactive"]

        print("✅ get_paginated_items with ExpressionAttributeNames test passed")

    finally:
        # Clean up the test items
        for item in test_items:
            ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, item["pk"], item["sk"])


def test_scan_for_items_with_expression_attribute_names(ddb: DynamoDB):
    """Test ExpressionAttributeNames support in scan_for_items method."""
    # Store test items with fields that could be reserved keywords
    test_items = [
        {"pk": "scan_test1", "sk": "item1", "name": "scan_item", "status": "active"},
        {"pk": "scan_test2", "sk": "item2", "name": "scan_item", "status": "inactive"},
    ]

    for item in test_items:
        ddb.store_item(TEST_TABLE_NAME, item)

    try:
        # Test scan_for_items with ExpressionAttributeNames
        scanned_items = ddb.scan_for_items(
            table_name=TEST_TABLE_NAME,
            filter_expression="contains(#n, :name_value)",
            expression_attribute_values={":name_value": {"S": "scan_item"}},
            projection_expression="pk, sk, #n, #s",
            expression_attribute_names={"#n": "name", "#s": "status"},
        )

        items_list = list(scanned_items)
        # Should find both items since both have name containing "scan_item"
        assert len(items_list) >= 2  # Could be more if other tests left items

        scan_test_items = [
            item for item in items_list if item["pk"].startswith("scan_test")
        ]
        assert len(scan_test_items) == 2

        for item in scan_test_items:
            expected_keys = {"pk", "sk", "name", "status"}
            actual_keys = set(item.keys())
            assert (
                actual_keys == expected_keys
            ), f"Expected keys {expected_keys}, got {actual_keys}"

            assert item["pk"] in ["scan_test1", "scan_test2"]
            assert item["sk"] in ["item1", "item2"]
            assert item["name"] == "scan_item"
            assert item["status"] in ["active", "inactive"]

        print("✅ scan_for_items with ExpressionAttributeNames test passed")

    finally:
        # Clean up the test items
        for item in test_items:
            ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, item["pk"], item["sk"])


def test_get_paginated_items_special_format(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk2", "sk": "sk2", "data": "data2"},
        {"pk": "pk3", "sk": "sk3", "data": b"binary_data"},
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    key_condition_expression = "pk = :pk"
    expression_attribute_values = {":pk": {"S": "pk1"}}

    paginated_items = ddb.get_paginated_items(
        TEST_TABLE_NAME, key_condition_expression, expression_attribute_values
    )

    paginated_items_list = list(paginated_items)
    assert len(paginated_items_list) == 1
    assert DynamoDB.item_to_dict(paginated_items_list[0]) == items_to_store[0]

    expression_attribute_values = {":pk": {"S": "pk3"}}
    paginated_items = ddb.get_paginated_items(
        TEST_TABLE_NAME, key_condition_expression, expression_attribute_values
    )

    paginated_items_list = list(paginated_items)

    assert len(paginated_items_list) == 1
    assert DynamoDB.item_to_dict(paginated_items_list[0]) == items_to_store[2]
    assert type(DynamoDB.item_to_dict(paginated_items_list[0])["data"].value) == bytes


@pytest.fixture
def ddb():
    return create_dynamo_db()


def create_dynamo_db():
    config = SimpleNamespace(REGION_NAME="eu-central-2")
    return DynamoDB(config)


def test_scan_for_items_by_pk_sk(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk2", "sk": "sk2", "data": "data2"},
        {"pk": "pk3", "sk": "sk3", "data": "data3"},
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    scanned_items = ddb.scan_for_items_by_pk_sk(
        TEST_TABLE_NAME, sk_contains="sk1", pk_contains=None
    )

    scanned_items_list = list(scanned_items)
    assert len(scanned_items_list) == 1
    assert scanned_items_list[0] == items_to_store[0]

    scanned_items = ddb.scan_for_items_by_pk_sk(
        TEST_TABLE_NAME, sk_contains="2", pk_contains="pk"
    )

    scanned_items_list = list(scanned_items)
    assert len(scanned_items_list) == 1
    assert scanned_items_list[0] == items_to_store[1]


def test_scan_for_items(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "content": "data1"},
        {"pk": "pk2", "sk": "sk2", "content": "data2"},
        {"pk": "pk2", "sk": "sk5", "content": "data2"},
        {"pk": "pk3", "sk": "sk3", "dacontenta": "data3"},
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    filter_expression = "contains(sk, :sk) and contains(content,:content)"
    expression_attribute_values = {":sk": {"S": "sk"}, ":content": {"S": "data2"}}

    scanned_items = ddb.scan_for_items(
        TEST_TABLE_NAME,
        filter_expression=filter_expression,
        expression_attribute_values=expression_attribute_values,
    )

    scanned_items_list = list(scanned_items)
    assert len(scanned_items_list) == 2
    assert scanned_items_list[0] == items_to_store[1]
    assert scanned_items_list[1] == items_to_store[2]

    expression_attribute_values = {":sk": {"S": "2"}, ":content": {"S": "data2"}}

    scanned_items = ddb.scan_for_items(
        TEST_TABLE_NAME,
        filter_expression=filter_expression,
        expression_attribute_values=expression_attribute_values,
    )

    scanned_items_list = list(scanned_items)
    assert len(scanned_items_list) == 1
    assert scanned_items_list[0] == items_to_store[1]


def test_get_paginated_items_by_pk(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk1", "sk": "sk2", "data": "data2"},
        {"pk": "pk1", "sk": "sk3", "data": "data3"},
        {"pk": "pk2", "sk": "sk3", "data": "data3"},
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    paginated_items = ddb.get_paginated_items_by_pk(TEST_TABLE_NAME, "pk1")

    paginated_items_list = list(paginated_items)
    assert len(paginated_items_list) == 3
    assert paginated_items_list == items_to_store[:3]

    paginated_items = ddb.get_paginated_items_by_pk(
        TEST_TABLE_NAME, "pk1", projection_expression="pk,sk"
    )

    paginated_items_list = list(paginated_items)
    assert len(paginated_items_list) == 3
    assert paginated_items_list == [
        {"pk": "pk1", "sk": "sk1"},
        {"pk": "pk1", "sk": "sk2"},
        {"pk": "pk1", "sk": "sk3"},
    ]


def test_query(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk1", "sk": "sk2", "data": "data2"},
        {"pk": "pk1", "sk": "sk3", "data": "data3"},
        {"pk": "pk2", "sk": "sk3", "data": "data3"},
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    query_result = ddb.query(
        TEST_TABLE_NAME,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": "pk1"}},
    )["Items"]

    assert len(query_result) == 3
    assert query_result == [DynamoDB.dict_to_item(i) for i in items_to_store[:3]]


def test_item_exists(ddb: DynamoDB):
    item_to_store = {"pk": "pk1", "sk": "sk1", "data": "data1"}
    ddb.store_item(TEST_TABLE_NAME, item_to_store)

    assert (
        ddb.item_exists(TEST_TABLE_NAME, item_to_store["pk"], item_to_store["sk"])
        == True
    )
    assert ddb.item_exists(TEST_TABLE_NAME, "pk7", "s1") == False
    assert ddb.item_exists(TEST_TABLE_NAME, "pk1", "sk7") == False
    assert ddb.item_exists(TEST_TABLE_NAME, "pk7", "sk7") == False


def test_batch_get_item(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk1", "sk": "sk2", "data": "data2"},
        {"pk": "pk1", "sk": "sk3", "data": "data3"},
        {"pk": "pk2", "sk": "sk3", "data": "data3"},
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    keys_to_fetch = [
        DynamoDB.dict_to_item({k: v for k, v in i.items() if k in ["pk", "sk"]})
        for i in items_to_store
    ]

    result = ddb.batch_get_item({TEST_TABLE_NAME: {"Keys": keys_to_fetch}})

    result["Responses"][TEST_TABLE_NAME] == [
        DynamoDB.dict_to_item(i) for i in items_to_store
    ]


def test_pk_sk_fields():
    pk = "GURK#1#QUARK#3"
    sk = "LALA#1#LULU#2"

    assert DynamoDB.pk_sk_fields(pk, sk) == {
        "gurk": "1",
        "quark": "3",
        "lala": "1",
        "lulu": "2",
    }


def test_item_to_dict():
    item = {"pk": {"S": "pk1"}, "sk": {"S": "sk1"}, "data": {"S": "data1"}}

    assert DynamoDB.item_to_dict(item) == {
        "pk": "pk1",
        "sk": "sk1",
        "data": "data1",
    }

    assert DynamoDB.item_to_dict(None) == {}


def test_batch_write(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk1", "sk": "sk2", "data": "data2"},
        {"pk": "pk1", "sk": "sk3", "data": "data3"},
        {"pk": "pk2", "sk": "sk3", "data": "data3"},
    ]

    ddb.batch_write(TEST_TABLE_NAME, items_to_store)

    for item in items_to_store:
        assert ddb.get_item_by_pk_sk(TEST_TABLE_NAME, item["pk"], item["sk"]) == item


def test_update_versioned_item(ddb: DynamoDB):
    item_to_store = {"pk": "pk7799", "sk": "sk2299", "version": 0, "data": "data1"}
    ddb.store_item(TEST_TABLE_NAME, item_to_store)

    ddb.update_versioned_item_by_pk_sk(
        TEST_TABLE_NAME,
        item_to_store["pk"],
        item_to_store["sk"],
        {"data": "data2"},
        expected_version=0,
    )

    assert ddb.get_item_by_pk_sk(
        TEST_TABLE_NAME, item_to_store["pk"], item_to_store["sk"]
    ) == {"pk": "pk7799", "sk": "sk2299", "version": 1, "data": "data2"}

    from poemai_utils.aws.dynamodb import VersionMismatchException

    with pytest.raises(VersionMismatchException):
        ddb.update_versioned_item_by_pk_sk(
            TEST_TABLE_NAME,
            item_to_store["pk"],
            item_to_store["sk"],
            {"data": "data3"},
            expected_version=0,
        )


def test_get_paginated_items_starting_at_pk_sk(ddb: DynamoDB):

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


def test_paginated_items_starting_at_pk_sk_sorting(ddb: DynamoDB):

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


def test_query_sort_key_pagination(ddb: DynamoDB):
    corpus_key = f"pagination_{uuid.uuid4().hex}"
    raw_content_ids = [
        "rci_0001",
        "rci_0002",
        "rci_0003",
        "rci_0004",
    ]

    items_to_store = [
        {"pk": corpus_key, "sk": raw_content_id, "raw_content_id": raw_content_id}
        for raw_content_id in raw_content_ids
    ]

    for item in items_to_store:
        ddb.store_item(TEST_TABLE_NAME, item)

    try:

        def fetch_page(after_sk=None, limit=2):
            key_condition_expression = "pk = :pk"
            expression_attribute_values = {":pk": {"S": corpus_key}}
            if after_sk is not None:
                key_condition_expression += " AND sk > :sk"
                expression_attribute_values[":sk"] = {"S": after_sk}

            page = []
            for item in ddb.get_paginated_items(
                table_name=TEST_TABLE_NAME,
                key_condition_expression=key_condition_expression,
                expression_attribute_values=expression_attribute_values,
                limit=limit,
            ):
                page.append(DynamoDB.item_to_dict(item)["sk"])
                if len(page) >= limit:
                    break
            return page

        first_page = fetch_page(limit=2)
        assert first_page == raw_content_ids[:2]

        second_page = fetch_page(after_sk=first_page[-1], limit=2)
        assert second_page == raw_content_ids[2:]

        empty_page = fetch_page(after_sk=raw_content_ids[-1], limit=2)
        assert empty_page == []
    finally:
        for item in items_to_store:
            ddb.delete_item_by_pk_sk(TEST_TABLE_NAME, item["pk"], item["sk"])
