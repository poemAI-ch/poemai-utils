"""
Unit tests for DynamoDB emulator index projection functionality.
"""

import pytest
from poemai_utils.aws.dynamodb import DynamoDB
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator


class TestDynamoDBEmulatorIndexProjection:
    """Test index projection enforcement in DynamoDB emulator."""

    @pytest.fixture
    def emulator_with_enforcement(self):
        """Create emulator with index enforcement enabled."""
        return DynamoDBEmulator(None, enforce_index_existence=True)

    @pytest.fixture
    def emulator_without_enforcement(self):
        """Create emulator without index enforcement."""
        return DynamoDBEmulator(None, enforce_index_existence=False)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return [
            {
                "pk": "CLIENT#client1",
                "sk": "BUTTON#button1#2023-01-01T00:00:00Z",
                "button_id": "button1",
                "site_id": "site1",
                "assigned_at": "2023-01-01T00:00:00Z",
                "extra_field": "value1",
            },
            {
                "pk": "CLIENT#client1",
                "sk": "BUTTON#button2#2023-01-02T00:00:00Z",
                "button_id": "button2",
                "site_id": "site2",
                "assigned_at": "2023-01-02T00:00:00Z",
                "extra_field": "value2",
            },
        ]

    def test_add_index_keys_only(self, emulator_with_enforcement):
        """Test adding a KEYS_ONLY index."""
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="KEYS_ONLY",
            hash_key="button_id",
            sort_key="assigned_at",
        )

        # Verify index was added
        assert "test_table" in emulator_with_enforcement.indexes
        assert (
            "button_id-assigned_at-index"
            in emulator_with_enforcement.indexes["test_table"]
        )

        index_spec = emulator_with_enforcement.indexes["test_table"][
            "button_id-assigned_at-index"
        ]
        assert index_spec["projection_type"] == "KEYS_ONLY"
        assert index_spec["hash_key"] == "button_id"
        assert index_spec["sort_key"] == "assigned_at"
        assert index_spec["non_key_attributes"] == []

    def test_add_index_include(self, emulator_with_enforcement):
        """Test adding an INCLUDE index."""
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="INCLUDE",
            hash_key="button_id",
            sort_key="assigned_at",
            non_key_attributes=["site_id"],
        )

        index_spec = emulator_with_enforcement.indexes["test_table"][
            "button_id-assigned_at-index"
        ]
        assert index_spec["projection_type"] == "INCLUDE"
        assert index_spec["non_key_attributes"] == ["site_id"]

    def test_add_index_all(self, emulator_with_enforcement):
        """Test adding an ALL index."""
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="ALL",
            hash_key="button_id",
            sort_key="assigned_at",
        )

        index_spec = emulator_with_enforcement.indexes["test_table"][
            "button_id-assigned_at-index"
        ]
        assert index_spec["projection_type"] == "ALL"

    def test_apply_index_projection_keys_only(self, emulator_with_enforcement):
        """Test KEYS_ONLY projection filtering."""
        # Add index
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="KEYS_ONLY",
            hash_key="button_id",
            sort_key="assigned_at",
        )

        # Test item
        item = {
            "pk": "CLIENT#client1",
            "sk": "BUTTON#button1#2023-01-01T00:00:00Z",
            "button_id": "button1",
            "assigned_at": "2023-01-01T00:00:00Z",
            "site_id": "site1",
            "extra_field": "value1",
        }

        # Apply projection
        projected = emulator_with_enforcement._apply_index_projection(
            item, "test_table", "button_id-assigned_at-index", ["pk", "sk"]
        )

        # Should only contain key attributes
        expected_keys = {"pk", "sk", "button_id", "assigned_at"}
        assert set(projected.keys()) == expected_keys
        assert projected["pk"] == "CLIENT#client1"
        assert projected["sk"] == "BUTTON#button1#2023-01-01T00:00:00Z"
        assert projected["button_id"] == "button1"
        assert projected["assigned_at"] == "2023-01-01T00:00:00Z"
        assert "site_id" not in projected
        assert "extra_field" not in projected

    def test_apply_index_projection_include(self, emulator_with_enforcement):
        """Test INCLUDE projection filtering."""
        # Add index
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="INCLUDE",
            hash_key="button_id",
            sort_key="assigned_at",
            non_key_attributes=["site_id"],
        )

        # Test item
        item = {
            "pk": "CLIENT#client1",
            "sk": "BUTTON#button1#2023-01-01T00:00:00Z",
            "button_id": "button1",
            "assigned_at": "2023-01-01T00:00:00Z",
            "site_id": "site1",
            "extra_field": "value1",
        }

        # Apply projection
        projected = emulator_with_enforcement._apply_index_projection(
            item, "test_table", "button_id-assigned_at-index", ["pk", "sk"]
        )

        # Should contain key attributes plus included attributes
        expected_keys = {"pk", "sk", "button_id", "assigned_at", "site_id"}
        assert set(projected.keys()) == expected_keys
        assert projected["site_id"] == "site1"
        assert "extra_field" not in projected

    def test_apply_index_projection_all(self, emulator_with_enforcement):
        """Test ALL projection (no filtering)."""
        # Add index
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="ALL",
            hash_key="button_id",
            sort_key="assigned_at",
        )

        # Test item
        item = {
            "pk": "CLIENT#client1",
            "sk": "BUTTON#button1#2023-01-01T00:00:00Z",
            "button_id": "button1",
            "assigned_at": "2023-01-01T00:00:00Z",
            "site_id": "site1",
            "extra_field": "value1",
        }

        # Apply projection
        projected = emulator_with_enforcement._apply_index_projection(
            item, "test_table", "button_id-assigned_at-index", ["pk", "sk"]
        )

        # Should contain all attributes
        assert projected == item

    def test_projection_without_enforcement(self, emulator_without_enforcement):
        """Test that projection is not applied when enforcement is disabled."""
        # Don't add any index definitions

        # Test item
        item = {
            "pk": "CLIENT#client1",
            "sk": "BUTTON#button1#2023-01-01T00:00:00Z",
            "button_id": "button1",
            "assigned_at": "2023-01-01T00:00:00Z",
            "site_id": "site1",
            "extra_field": "value1",
        }

        # Apply projection (should return original item)
        projected = emulator_without_enforcement._apply_index_projection(
            item, "test_table", "button_id-assigned_at-index", ["pk", "sk"]
        )

        # Should return the original item unchanged
        assert projected == item

    def test_projection_with_missing_index(self, emulator_with_enforcement):
        """Test that projection returns original item when index is not defined."""
        # Don't add any index definitions

        # Test item
        item = {
            "pk": "CLIENT#client1",
            "sk": "BUTTON#button1#2023-01-01T00:00:00Z",
            "button_id": "button1",
            "assigned_at": "2023-01-01T00:00:00Z",
            "site_id": "site1",
            "extra_field": "value1",
        }

        # Apply projection for undefined index
        projected = emulator_with_enforcement._apply_index_projection(
            item, "test_table", "undefined_index", ["pk", "sk"]
        )

        # Should return the original item unchanged
        assert projected == item

    def test_get_paginated_items_with_index_enforcement(
        self, emulator_with_enforcement, sample_data
    ):
        """Test that get_paginated_items enforces index existence."""
        # Add sample data
        for item in sample_data:
            emulator_with_enforcement.store_item("test_table", item)

        # Test with undefined index - should raise error
        with pytest.raises(ValueError, match="Index undefined_index not found"):
            list(
                emulator_with_enforcement.get_paginated_items(
                    table_name="test_table",
                    key_condition_expression="pk = :pk",
                    expression_attribute_values={":pk": {"S": "CLIENT#client1"}},
                    index_name="undefined_index",
                )
            )

    def test_get_paginated_items_with_projection(
        self, emulator_with_enforcement, sample_data
    ):
        """Test that get_paginated_items applies projection correctly."""
        # Add index
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="button_id-assigned_at-index",
            projection_type="KEYS_ONLY",
            hash_key="button_id",
            sort_key="assigned_at",
        )

        # Add sample data
        for item in sample_data:
            emulator_with_enforcement.store_item("test_table", item)

        # Query with index
        results = list(
            emulator_with_enforcement.get_paginated_items(
                table_name="test_table",
                key_condition_expression="pk = :pk",
                expression_attribute_values={":pk": {"S": "CLIENT#client1"}},
                index_name="button_id-assigned_at-index",
            )
        )

        # Verify projection was applied
        assert len(results) == 2
        for result in results:
            item_dict = DynamoDB.item_to_dict(result)
            # Should only contain key attributes
            expected_keys = {"pk", "sk", "button_id", "assigned_at"}
            assert set(item_dict.keys()) == expected_keys
            assert "site_id" not in item_dict
            assert "extra_field" not in item_dict

    def test_get_paginated_items_without_index(
        self, emulator_with_enforcement, sample_data
    ):
        """Test that get_paginated_items works normally without index."""
        # Add sample data
        for item in sample_data:
            emulator_with_enforcement.store_item("test_table", item)

        # Query without index
        results = list(
            emulator_with_enforcement.get_paginated_items(
                table_name="test_table",
                key_condition_expression="pk = :pk",
                expression_attribute_values={":pk": {"S": "CLIENT#client1"}},
            )
        )

        # Should return all attributes (no projection)
        assert len(results) == 2
        for result in results:
            item_dict = DynamoDB.item_to_dict(result)
            # Should contain all attributes
            assert "site_id" in item_dict
            assert "extra_field" in item_dict

    def test_projection_expression_with_missing_fields(self, emulator_with_enforcement):
        """Test that projection expression properly handles items missing requested fields."""

        # Add index for corpus_key -> raw_content_id (mimicking our real scenario)
        emulator_with_enforcement.add_index(
            table_name="test_table",
            index_name="corpus_key-raw_content_id-index",
            projection_type="KEYS_ONLY",
            hash_key="corpus_key",
            sort_key="raw_content_id",
        )

        # Store two types of items with same corpus_key:
        # 1. CORPUS_METADATA item (has corpus_key but NO raw_content_id)
        corpus_metadata_item = {
            "pk": "CORPUS_METADATA#",
            "sk": "CORPUS_KEY#test_corpus",
            "corpus_key": "test_corpus",
            "model": "test-model",
            # Note: NO raw_content_id field
        }
        emulator_with_enforcement.store_item("test_table", corpus_metadata_item)

        # 2. METADATA item (has both corpus_key AND raw_content_id)
        metadata_item = {
            "pk": "METADATA#",
            "sk": "RAW_CONTENT_ID#abc123",
            "corpus_key": "test_corpus",
            "raw_content_id": "abc123",
            "title": "Test Document",
        }
        emulator_with_enforcement.store_item("test_table", metadata_item)

        # Query with projection expression that requests both pk and raw_content_id
        results = list(
            emulator_with_enforcement.get_paginated_items(
                table_name="test_table",
                key_condition_expression="corpus_key = :corpus_key",
                expression_attribute_values={":corpus_key": {"S": "test_corpus"}},
                projection_expression="pk,raw_content_id",  # This should filter out items without raw_content_id
                index_name="corpus_key-raw_content_id-index",
            )
        )

        # EXPECTED: Only 1 result (the METADATA item that has raw_content_id)
        # ACTUAL BUG: 2 results including CORPUS_METADATA item without raw_content_id

        print(f"Number of results: {len(results)}")
        for i, result in enumerate(results):
            item_dict = DynamoDB.item_to_dict(result)
            print(f"Result {i}: {item_dict}")

        # This assertion should pass but currently fails due to the bug
        assert (
            len(results) == 1
        ), f"Expected 1 result, got {len(results)}: projection should filter out items missing raw_content_id"

        # The single result should be the METADATA item
        result_item = DynamoDB.item_to_dict(results[0])
        assert result_item["pk"] == "METADATA#"
        assert "raw_content_id" in result_item
        assert result_item["raw_content_id"] == "abc123"

    def test_query_range_operators_skip_missing_sort_keys(
        self, emulator_with_enforcement
    ):
        """Test range operators with missing attributes in KeyConditionExpression."""
        table_name = "test_table"
        corpus_key = "test_corpus"

        emulator_with_enforcement.add_index(
            table_name=table_name,
            index_name="corpus_key-raw_content_id-index",
            projection_type="KEYS_ONLY",
            hash_key="corpus_key",
            sort_key="raw_content_id",
        )

        items = [
            {
                "pk": "METADATA#",
                "sk": "RAW_CONTENT_ID#rci_0001",
                "corpus_key": corpus_key,
                "raw_content_id": "rci_0001",
            },
            {
                "pk": "METADATA#",
                "sk": "RAW_CONTENT_ID#rci_0002",
                "corpus_key": corpus_key,
                "raw_content_id": "rci_0002",
            },
            {
                "pk": "METADATA#",
                "sk": "RAW_CONTENT_ID#rci_0003",
                "corpus_key": corpus_key,
                "raw_content_id": "rci_0003",
            },
            {
                "pk": "CORPUS_METADATA#",
                "sk": "CORPUS_KEY#test_corpus",
                "corpus_key": corpus_key,
            },
        ]

        for item in items:
            emulator_with_enforcement.store_item(table_name, item)

        base_query_args = {
            "table_name": table_name,
            "projection_expression": "pk,raw_content_id",
            "index_name": "corpus_key-raw_content_id-index",
        }

        results_gt = list(
            emulator_with_enforcement.get_paginated_items(
                key_condition_expression="corpus_key = :corpus_key AND raw_content_id > :raw_content_id",
                expression_attribute_values={
                    ":corpus_key": {"S": corpus_key},
                    ":raw_content_id": {"S": "rci_0001"},
                },
                **base_query_args,
            )
        )
        raw_content_ids_gt = [
            DynamoDB.item_to_dict(item)["raw_content_id"] for item in results_gt
        ]
        assert raw_content_ids_gt == ["rci_0002", "rci_0003"]

        results_lt = list(
            emulator_with_enforcement.get_paginated_items(
                key_condition_expression="corpus_key = :corpus_key AND raw_content_id < :raw_content_id",
                expression_attribute_values={
                    ":corpus_key": {"S": corpus_key},
                    ":raw_content_id": {"S": "rci_0003"},
                },
                **base_query_args,
            )
        )
        raw_content_ids_lt = [
            DynamoDB.item_to_dict(item)["raw_content_id"] for item in results_lt
        ]
        assert raw_content_ids_lt == ["rci_0001", "rci_0002"]
