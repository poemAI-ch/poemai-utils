import pytest
from botocore.exceptions import ClientError
from poemai_utils.aws.dynamodb import DynamoDB
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator


def _item(values):
    return DynamoDB.dict_to_item(values)


def test_emulator_transact_write_items_commits_supported_operations():
    db = DynamoDBEmulator(
        None,
        allowed_reserved_keywords=["active", "state", "value", "version"],
    )
    table_name = "messaging"
    db.store_item(
        table_name,
        {"pk": "route", "sk": "current", "version": 0, "state": "open"},
    )
    db.store_item(table_name, {"pk": "old", "sk": "item", "value": "remove"})
    db.store_item(table_name, {"pk": "config", "sk": "active", "active": True})

    result = db.transact_write_items(
        [
            {
                "Put": {
                    "TableName": table_name,
                    "Item": _item(
                        {"pk": "event", "sk": "provider-1", "status": "completed"}
                    ),
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
            {
                "Update": {
                    "TableName": table_name,
                    "Key": _item({"pk": "route", "sk": "current"}),
                    "UpdateExpression": "SET #version = :new_version, #state = :closed",
                    "ConditionExpression": (
                        "#version = :expected_version AND #state = :open"
                    ),
                    "ExpressionAttributeNames": {
                        "#version": "version",
                        "#state": "state",
                    },
                    "ExpressionAttributeValues": {
                        ":new_version": {"N": "1"},
                        ":expected_version": {"N": "0"},
                        ":closed": {"S": "closed"},
                        ":open": {"S": "open"},
                    },
                }
            },
            {
                "Delete": {
                    "TableName": table_name,
                    "Key": _item({"pk": "old", "sk": "item"}),
                    "ConditionExpression": "attribute_exists(pk)",
                }
            },
            {
                "ConditionCheck": {
                    "TableName": table_name,
                    "Key": _item({"pk": "config", "sk": "active"}),
                    "ConditionExpression": "#active = :active",
                    "ExpressionAttributeNames": {"#active": "active"},
                    "ExpressionAttributeValues": {":active": {"BOOL": True}},
                }
            },
        ],
        ClientRequestToken="stable-request-token-1234567890",
    )

    assert result["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert db.get_item_by_pk_sk(table_name, "event", "provider-1") == {
        "pk": "event",
        "sk": "provider-1",
        "status": "completed",
    }
    assert db.get_item_by_pk_sk(table_name, "route", "current") == {
        "pk": "route",
        "sk": "current",
        "version": 1,
        "state": "closed",
    }
    assert db.get_item_by_pk_sk(table_name, "old", "item") is None
    assert db.get_item_by_pk_sk(table_name, "config", "active")["active"] is True


def test_emulator_transact_write_items_rolls_back_on_condition_failure():
    db = DynamoDBEmulator(
        None,
        allowed_reserved_keywords=["state", "version"],
    )
    table_name = "messaging"
    db.store_item(
        table_name,
        {"pk": "route", "sk": "current", "version": 2, "state": "open"},
    )

    with pytest.raises(ClientError) as exc_info:
        db.transact_write_items(
            [
                {
                    "Put": {
                        "TableName": table_name,
                        "Item": _item(
                            {"pk": "event", "sk": "provider-1", "status": "new"}
                        ),
                        "ConditionExpression": "attribute_not_exists(pk)",
                    }
                },
                {
                    "Update": {
                        "TableName": table_name,
                        "Key": _item({"pk": "route", "sk": "current"}),
                        "UpdateExpression": "SET #state = :closed",
                        "ConditionExpression": "#version = :expected_version",
                        "ExpressionAttributeNames": {
                            "#state": "state",
                            "#version": "version",
                        },
                        "ExpressionAttributeValues": {
                            ":closed": {"S": "closed"},
                            ":expected_version": {"N": "1"},
                        },
                    }
                },
            ]
        )

    assert exc_info.value.response["Error"]["Code"] == "TransactionCanceledException"
    assert exc_info.value.response["CancellationReasons"][1]["Code"] == (
        "ConditionalCheckFailed"
    )
    assert db.get_item_by_pk_sk(table_name, "event", "provider-1") is None
    assert db.get_item_by_pk_sk(table_name, "route", "current") == {
        "pk": "route",
        "sk": "current",
        "version": 2,
        "state": "open",
    }


def test_emulator_consistent_read_bypasses_stale_read_simulation():
    db = DynamoDBEmulator(
        None,
        eventual_consistency_config={
            "enabled": True,
            "delay_reads": 2,
            "patterns": [{"table_name": "messaging", "pk": "route"}],
        },
        allowed_reserved_keywords=["state"],
    )
    db.store_item(
        "messaging",
        {"pk": "route", "sk": "current", "state": "active"},
    )

    assert db.get_item_by_pk_sk("messaging", "route", "current") is None
    assert db.get_item_by_pk_sk(
        "messaging",
        "route",
        "current",
        consistent_read=True,
    ) == {"pk": "route", "sk": "current", "state": "active"}
    assert db.get_item_by_pk_sk("messaging", "route", "current") is None
