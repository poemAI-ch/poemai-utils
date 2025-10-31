import logging
from enum import Enum
from unittest.mock import MagicMock

import pytest
from poemai_utils.aws.dao_helper import DaoHelper
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator
from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr

_logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "openai.text-embedding-ada-002"


class KeyElement(str, Enum):
    CORPUS_KEY = "CORPUS_KEY"
    RAW_CONTENT_ID = "RAW_CONTENT_ID"
    PAGE = "PAGE"
    PARAGRAPH_NR = "PARAGRAPH_NR"
    OBJECT_TYPE = "OBJECT_TYPE"


add_enum_repr(KeyElement)


FIELD_TO_KEY_FORMATTERS = {
    KeyElement.PAGE: {
        "formatter": lambda x: str(int(x)).zfill(4),
    },
    KeyElement.PARAGRAPH_NR: {
        "formatter": lambda x: str(int(x)).zfill(6),
    },
}

KEY_TO_FIELD_FORMATTERS = {
    KeyElement.PAGE: {
        "formatter": lambda x: int(x),
    },
    KeyElement.PARAGRAPH_NR: {
        "formatter": lambda x: int(x),
    },
}


class ObjectTypeKeys(str, Enum):

    DOCUMENT = "DOCUMENT"
    PARAGRAPH = "PARAGRAPH"


add_enum_repr(ObjectTypeKeys)


add_enum_attrs(
    {
        ObjectTypeKeys.PARAGRAPH: {
            "pk_components": [KeyElement.RAW_CONTENT_ID],
            "sk_components": [
                KeyElement.OBJECT_TYPE,
                KeyElement.PAGE,
                KeyElement.PARAGRAPH_NR,
            ],
            "required_fields": [
                KeyElement.RAW_CONTENT_ID,
                KeyElement.PAGE,
                KeyElement.PARAGRAPH_NR,
            ],
            "to_drop_fields": [
                KeyElement.OBJECT_TYPE,
                KeyElement.PAGE,
                KeyElement.PARAGRAPH_NR,
                KeyElement.RAW_CONTENT_ID,
            ],
        },
        ObjectTypeKeys.DOCUMENT: {
            "pk_components": [KeyElement.OBJECT_TYPE],
            "sk_components": [KeyElement.RAW_CONTENT_ID],
            "required_fields": [KeyElement.RAW_CONTENT_ID, KeyElement.CORPUS_KEY],
            "to_drop_fields": [KeyElement.OBJECT_TYPE],
        },
    }
)


@pytest.fixture
def dynamo_db():
    db = DynamoDBEmulator(None)

    return db


def test_store_object(dynamo_db):
    # Create a mock DB object
    dynamo_db = MagicMock()
    table_name = "test_table"

    # Test storing a regular document
    values = {
        "raw_content_id": "doc456",
        "corpus_key": "corpus789",
        "title": "Test Document",
        "content": "This is test content",
    }

    result = DaoHelper.store_object(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        dynamo_db,
        table_name,
        ObjectTypeKeys.DOCUMENT,
        values,
    )

    # Verify the result contains the correct keys
    assert result["pk"] == "DOCUMENT#"
    assert result["sk"] == "RAW_CONTENT_ID#doc456"

    # Verify the mock was called with the correct parameters
    expected_item = {
        "raw_content_id": "doc456",
        "corpus_key": "corpus789",
        "title": "Test Document",
        "content": "This is test content",
        "pk": "DOCUMENT#",
        "sk": "RAW_CONTENT_ID#doc456",
    }
    dynamo_db.store_item.assert_called_once_with(table_name, expected_item)

    # Test storing a versioned object
    dynamo_db.reset_mock()
    values = {
        "raw_content_id": "doc789",
        "corpus_key": "corpus123",
        "title": "Versioned Document",
        "content": "This is versioned content",
        "version": 3,
    }

    result = DaoHelper.store_object(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        dynamo_db,
        table_name,
        ObjectTypeKeys.DOCUMENT,
        values,
        versioned=True,
    )

    # Verify the result contains the correct keys
    assert result["pk"] == "DOCUMENT#"
    assert result["sk"] == "RAW_CONTENT_ID#doc789"

    # Verify the mock was called with the correct parameters
    expected_values = {
        "raw_content_id": "doc789",
        "corpus_key": "corpus123",
        "title": "Versioned Document",
        "content": "This is versioned content",
    }
    dynamo_db.update_versioned_item_by_pk_sk.assert_called_once_with(
        table_name, "DOCUMENT#", "RAW_CONTENT_ID#doc789", expected_values, 3
    )

    # Test storing an object with only_if_new=True
    dynamo_db.reset_mock()
    values = {
        "raw_content_id": "doc101",
        "corpus_key": "corpus202",
        "title": "New Document",
        "content": "This is new content",
    }

    result = DaoHelper.store_object(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        dynamo_db,
        table_name,
        ObjectTypeKeys.DOCUMENT,
        values,
        only_if_new=True,
    )

    # Verify the result contains the correct keys
    assert result["pk"] == "DOCUMENT#"
    assert result["sk"] == "RAW_CONTENT_ID#doc101"

    # Verify the mock was called with the correct parameters
    expected_item = {
        "raw_content_id": "doc101",
        "corpus_key": "corpus202",
        "title": "New Document",
        "content": "This is new content",
        "pk": "DOCUMENT#",
        "sk": "RAW_CONTENT_ID#doc101",
    }
    dynamo_db.store_new_item.assert_called_once_with(table_name, expected_item, "sk")
