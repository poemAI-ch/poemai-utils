"""
Unit tests for DynamoDB class focusing on update_versioned_item method.

These tests specifically cover:
1. ConditionalCheckFailedException handling (version mismatch)
2. Expression size validation (> 4KB limit)
3. Successful update scenarios
4. ClientError import and exception handling
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from poemai_utils.aws.dynamodb import DynamoDB, VersionMismatchException

_logger = logging.getLogger(__name__)


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    return SimpleNamespace(REGION_NAME="us-east-1")


@pytest.fixture
def mock_dynamodb_client():
    """Create a mock boto3 DynamoDB client."""
    return MagicMock()


@pytest.fixture
def dynamodb(mock_config, mock_dynamodb_client):
    """Create a DynamoDB instance with a mock client."""
    db = DynamoDB(config=mock_config, dynamodb_client=mock_dynamodb_client)
    return db


def test_update_versioned_item_success(dynamodb, mock_dynamodb_client):
    """Test successful versioned item update."""
    # Arrange
    mock_dynamodb_client.update_item.return_value = {
        "Attributes": {"version": {"N": "2"}},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }

    # Act
    result = dynamodb.update_versioned_item(
        dynamodb_client=mock_dynamodb_client,
        table_name="test_table",
        hash_key="test_pk",
        hash_key_name="pk",
        attribute_updates={"data": "new_value"},
        expected_version=1,
        version_attribute_name="version",
        range_key="test_sk",
        range_key_name="sk",
    )

    # Assert
    assert result is not None
    mock_dynamodb_client.update_item.assert_called_once()
    call_kwargs = mock_dynamodb_client.update_item.call_args[1]
    assert call_kwargs["TableName"] == "test_table"
    assert call_kwargs["Key"] == {"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}}
    assert "SET" in call_kwargs["UpdateExpression"]
    assert call_kwargs["ConditionExpression"] == "#version = :expectedVersion"


def test_update_versioned_item_conditional_check_failed(dynamodb, mock_dynamodb_client):
    """Test that ConditionalCheckFailedException is properly caught and converted to VersionMismatchException.

    This test verifies the fix for the ClientError import scoping issue.
    """
    # Arrange - create a proper ClientError with ConditionalCheckFailedException
    error_response = {
        "Error": {
            "Code": "ConditionalCheckFailedException",
            "Message": "The conditional request failed",
        },
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    mock_dynamodb_client.update_item.side_effect = ClientError(
        error_response, "UpdateItem"
    )

    # Act & Assert
    with pytest.raises(VersionMismatchException) as exc_info:
        dynamodb.update_versioned_item(
            dynamodb_client=mock_dynamodb_client,
            table_name="test_table",
            hash_key="test_pk",
            hash_key_name="pk",
            attribute_updates={"data": "new_value"},
            expected_version=1,
            version_attribute_name="version",
            range_key="test_sk",
            range_key_name="sk",
        )

    # Verify the exception message contains expected information
    assert "Version mismatch" in str(exc_info.value)
    assert "expecting 1" in str(exc_info.value)

    # Verify that the original ClientError is chained
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ClientError)


def test_update_versioned_item_other_client_error(dynamodb, mock_dynamodb_client):
    """Test that non-ConditionalCheckFailedException ClientErrors are re-raised."""
    # Arrange
    error_response = {
        "Error": {
            "Code": "ResourceNotFoundException",
            "Message": "Table not found",
        },
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    mock_dynamodb_client.update_item.side_effect = ClientError(
        error_response, "UpdateItem"
    )

    # Act & Assert
    with pytest.raises(ClientError) as exc_info:
        dynamodb.update_versioned_item(
            dynamodb_client=mock_dynamodb_client,
            table_name="test_table",
            hash_key="test_pk",
            hash_key_name="pk",
            attribute_updates={"data": "new_value"},
            expected_version=1,
            version_attribute_name="version",
        )

    # Verify it's the original ClientError, not wrapped
    assert exc_info.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_update_versioned_item_expression_size_validation(mock_config):
    """Test that oversized expressions are rejected before sending to DynamoDB."""
    # Arrange - create a large attribute update that exceeds 4KB limit
    mock_client = MagicMock()
    dynamodb = DynamoDB(config=mock_config, dynamodb_client=mock_client)

    # Create many large attribute updates to exceed 4KB
    large_attribute_updates = {}
    for i in range(200):
        # Each attribute name contributes to the expression size
        attr_name = f"very_long_attribute_name_to_increase_size_{i:04d}"
        large_attribute_updates[attr_name] = "x" * 100

    # Act & Assert
    with pytest.raises(ClientError) as exc_info:
        dynamodb.update_versioned_item(
            dynamodb_client=mock_client,
            table_name="test_table",
            hash_key="test_pk",
            hash_key_name="pk",
            attribute_updates=large_attribute_updates,
            expected_version=1,
            version_attribute_name="version",
        )

    # Verify the error details
    assert exc_info.value.response["Error"]["Code"] == "ValidationException"
    assert (
        "Expression size has exceeded the maximum allowed size"
        in exc_info.value.response["Error"]["Message"]
    )
    assert "4096 bytes" in exc_info.value.response["Error"]["Message"]

    # Verify that update_item was never called (validation failed before API call)
    mock_client.update_item.assert_not_called()


def test_update_versioned_item_by_pk_sk_delegates_correctly(
    dynamodb, mock_dynamodb_client
):
    """Test that update_versioned_item_by_pk_sk properly delegates to update_versioned_item."""
    # Arrange
    mock_dynamodb_client.update_item.return_value = {
        "Attributes": {"version": {"N": "2"}},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }

    # Make sure the dynamodb instance uses our mock client
    dynamodb.dynamodb_client = mock_dynamodb_client

    # Act
    result = dynamodb.update_versioned_item_by_pk_sk(
        table_name="test_table",
        pk="test_pk",
        sk="test_sk",
        attribute_updates={"data": "new_value"},
        expected_version=1,
        version_attribute_name="custom_version",
    )

    # Assert
    assert result is not None
    mock_dynamodb_client.update_item.assert_called_once()
    call_kwargs = mock_dynamodb_client.update_item.call_args[1]
    assert call_kwargs["Key"] == {"pk": {"S": "test_pk"}, "sk": {"S": "test_sk"}}
    # Verify custom version attribute name was used
    assert "#custom_version" in call_kwargs["ExpressionAttributeNames"]


def test_update_versioned_item_no_range_key(dynamodb, mock_dynamodb_client):
    """Test update with hash key only (no range key)."""
    # Arrange
    mock_dynamodb_client.update_item.return_value = {
        "Attributes": {"version": {"N": "2"}},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }

    # Act
    result = dynamodb.update_versioned_item(
        dynamodb_client=mock_dynamodb_client,
        table_name="test_table",
        hash_key="test_pk",
        hash_key_name="pk",
        attribute_updates={"data": "new_value"},
        expected_version=1,
        version_attribute_name="version",
        range_key=None,
        range_key_name=None,
    )

    # Assert
    assert result is not None
    call_kwargs = mock_dynamodb_client.update_item.call_args[1]
    # Should only have hash key, not range key
    assert call_kwargs["Key"] == {"pk": {"S": "test_pk"}}
    assert "sk" not in call_kwargs["Key"]


def test_update_versioned_item_version_increment(dynamodb, mock_dynamodb_client):
    """Test that version is properly incremented in the update expression."""
    # Arrange
    mock_dynamodb_client.update_item.return_value = {
        "Attributes": {"version": {"N": "6"}},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }

    # Act
    dynamodb.update_versioned_item(
        dynamodb_client=mock_dynamodb_client,
        table_name="test_table",
        hash_key="test_pk",
        hash_key_name="pk",
        attribute_updates={"data": "new_value"},
        expected_version=5,  # Current version
        version_attribute_name="version",
    )

    # Assert
    call_kwargs = mock_dynamodb_client.update_item.call_args[1]
    # Verify the new version is set to expected_version + 1
    assert call_kwargs["ExpressionAttributeValues"][":newVersion"] == {"N": "6"}
    assert call_kwargs["ExpressionAttributeValues"][":expectedVersion"] == {"N": "5"}


def test_update_versioned_item_multiple_attributes(dynamodb, mock_dynamodb_client):
    """Test updating multiple attributes at once."""
    # Arrange
    mock_dynamodb_client.update_item.return_value = {
        "Attributes": {"version": {"N": "2"}},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }

    # Act
    dynamodb.update_versioned_item(
        dynamodb_client=mock_dynamodb_client,
        table_name="test_table",
        hash_key="test_pk",
        hash_key_name="pk",
        attribute_updates={
            "data": "new_value",
            "count": 42,
            "active": True,
            "tags": ["tag1", "tag2"],
        },
        expected_version=1,
        version_attribute_name="version",
    )

    # Assert
    call_kwargs = mock_dynamodb_client.update_item.call_args[1]
    # Verify all attributes are in the expression
    assert "#data" in call_kwargs["ExpressionAttributeNames"]
    assert "#count" in call_kwargs["ExpressionAttributeNames"]
    assert "#active" in call_kwargs["ExpressionAttributeNames"]
    assert "#tags" in call_kwargs["ExpressionAttributeNames"]
    # Verify all attributes have values
    assert ":data" in call_kwargs["ExpressionAttributeValues"]
    assert ":count" in call_kwargs["ExpressionAttributeValues"]
    assert ":active" in call_kwargs["ExpressionAttributeValues"]
    assert ":tags" in call_kwargs["ExpressionAttributeValues"]


def test_client_error_is_properly_imported():
    """Test that ClientError can be caught without scoping issues.

    This test specifically validates the fix for the import scoping bug
    where ClientError was imported inside an if block.
    """
    # This test verifies that ClientError is available at module level
    from botocore.exceptions import ClientError as BotocoreClientError
    from poemai_utils.aws.dynamodb import ClientError as ImportedClientError

    # They should be the same class
    assert ImportedClientError is BotocoreClientError

    # And we should be able to instantiate it
    error_response = {
        "Error": {"Code": "TestError", "Message": "Test message"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    error = ImportedClientError(error_response, "TestOperation")
    assert error.response["Error"]["Code"] == "TestError"
