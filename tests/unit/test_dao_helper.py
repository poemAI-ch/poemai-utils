import logging
from enum import Enum
from unittest.mock import MagicMock

from poemai_utils.aws.dao_helper import DaoHelper
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


def test_build_key():

    key = DaoHelper.build_key(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        [KeyElement.CORPUS_KEY],
        {"corpus_key": "test"},
    )

    assert key == "CORPUS_KEY#test"


def test_build_pk_sk():
    # Test for PARAGRAPH object type
    values = {
        "raw_content_id": "doc123",
        "page": 5,
        "paragraph_nr": 42,
    }

    pk, sk = DaoHelper.build_pk_sk(
        KeyElement, FIELD_TO_KEY_FORMATTERS, ObjectTypeKeys.PARAGRAPH, values
    )

    assert pk == "RAW_CONTENT_ID#doc123"
    assert sk == "PARAGRAPH##PAGE#0005#PARAGRAPH_NR#000042"

    # Test for DOCUMENT object type
    values = {
        "raw_content_id": "doc456",
        "corpus_key": "corpus789",
    }

    pk, sk = DaoHelper.build_pk_sk(
        KeyElement, FIELD_TO_KEY_FORMATTERS, ObjectTypeKeys.DOCUMENT, values
    )

    assert pk == "DOCUMENT#"
    assert sk == "RAW_CONTENT_ID#doc456"


def test_store_object():
    # Create a mock DB object
    mock_db = MagicMock()
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
        mock_db,
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
    mock_db.store_item.assert_called_once_with(table_name, expected_item)

    # Test storing a versioned object
    mock_db.reset_mock()
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
        mock_db,
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
    mock_db.update_versioned_item_by_pk_sk.assert_called_once_with(
        table_name, "DOCUMENT#", "RAW_CONTENT_ID#doc789", expected_values, 3
    )

    # Test storing an object with only_if_new=True
    mock_db.reset_mock()
    values = {
        "raw_content_id": "doc101",
        "corpus_key": "corpus202",
        "title": "New Document",
        "content": "This is new content",
    }

    result = DaoHelper.store_object(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        mock_db,
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
    mock_db.store_new_item.assert_called_once_with(table_name, expected_item, "sk")


def test_get_object_with_pk_sk_fields():
    # Create a mock DB object
    mock_db = MagicMock()
    table_name = "test_table"

    # Test with PARAGRAPH object type
    # First, define values for a paragraph object
    values = {
        "raw_content_id": "doc123",
        "page": 5,
        "paragraph_nr": 42,
        "content": "This is paragraph content",
        "embedding": [0.1, 0.2, 0.3],
    }

    # Mock what would be returned from the database
    # Note that page, paragraph_nr, and raw_content_id are not stored in the item
    # because they're in the to_drop_fields list
    db_returned_item = {
        "pk": "RAW_CONTENT_ID#doc123",
        "sk": "PARAGRAPH##PAGE#0005#PARAGRAPH_NR#000042",
        "content": "This is paragraph content",
        "embedding": [0.1, 0.2, 0.3],
    }

    # Set up the mock to return our item
    mock_db.get_item_by_pk_sk.return_value = db_returned_item

    # Call get_object_with_pk_sk_fields with the KEY_ELEMENT_ITEM_FORMATTERS
    result = DaoHelper.get_object_with_pk_sk_fields(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        KEY_TO_FIELD_FORMATTERS,
        mock_db,
        table_name,
        ObjectTypeKeys.PARAGRAPH,
        values,
    )

    # Verify the mock was called with the correct parameters
    mock_db.get_item_by_pk_sk.assert_called_once_with(
        table_name, "RAW_CONTENT_ID#doc123", "PARAGRAPH##PAGE#0005#PARAGRAPH_NR#000042"
    )

    # Verify the result contains both the original item and the reconstructed fields
    assert result["pk"] == "RAW_CONTENT_ID#doc123"
    assert result["sk"] == "PARAGRAPH##PAGE#0005#PARAGRAPH_NR#000042"
    assert result["content"] == "This is paragraph content"
    assert result["embedding"] == [0.1, 0.2, 0.3]

    # Verify the fields that were dropped are now reconstructed from the keys
    assert result["raw_content_id"] == "doc123"
    assert result["page"] == 5  # Converted back from "0005"
    assert result["paragraph_nr"] == 42  # Converted back from "000042"

    # Test with a non-existent item
    mock_db.reset_mock()
    mock_db.get_item_by_pk_sk.return_value = None

    result = DaoHelper.get_object_with_pk_sk_fields(
        KeyElement,
        FIELD_TO_KEY_FORMATTERS,
        KEY_TO_FIELD_FORMATTERS,  # Use the item formatters here
        mock_db,
        table_name,
        ObjectTypeKeys.PARAGRAPH,
        values,
    )

    assert result is None
