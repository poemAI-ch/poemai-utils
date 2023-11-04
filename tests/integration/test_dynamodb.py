import logging
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


def test_get_paginated_items_special_format(ddb: DynamoDB):
    items_to_store = [
        {"pk": "pk1", "sk": "sk1", "data": "data1"},
        {"pk": "pk2", "sk": "sk2", "data": "data2"},
        {"pk": "pk3", "sk": "sk3", "data": "data3"},
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
