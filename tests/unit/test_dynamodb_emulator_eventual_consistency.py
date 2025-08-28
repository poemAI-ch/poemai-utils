"""
Unit tests for the eventual consistency simulator in DynamoDBEmulator.
"""

import pytest
from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator


class TestDynamoDBEmulatorEventualConsistency:
    """Test eventual consistency simulation functionality."""

    def test_eventual_consistency_disabled_by_default(self):
        """Test that eventual consistency simulation is disabled by default."""
        emulator = DynamoDBEmulator(sqlite_filename=None)

        # Store an initial item
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "old_value"}
        )

        # Update the item
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "new_value"}
        )

        # Should immediately return the new value (no eventual consistency)
        result = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result["data"] == "new_value"

    def test_eventual_consistency_with_pattern_matching(self):
        """Test eventual consistency simulation with pattern matching."""
        config = {
            "enabled": True,
            "delay_reads": 2,
            "patterns": [
                {"table_name": "test_table", "pk": "test_pk", "sk": "test_sk"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None, eventual_consistency_config=config
        )

        # Store an initial item
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "old_value"}
        )

        # Update the item - this should trigger eventual consistency simulation
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "new_value"}
        )

        # First read should return stale data
        result1 = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result1["data"] == "old_value"

        # Second read should still return stale data
        result2 = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result2["data"] == "old_value"

        # Third read should return fresh data
        result3 = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result3["data"] == "new_value"

    def test_eventual_consistency_new_item_creation(self):
        """Test eventual consistency simulation when creating a new item."""
        config = {
            "enabled": True,
            "delay_reads": 1,
            "patterns": [{"table_name": "test_table", "pk": "new_item"}],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None, eventual_consistency_config=config
        )

        # Create a new item - this should trigger eventual consistency simulation
        emulator.store_item(
            "test_table", {"pk": "new_item", "sk": "test_sk", "data": "new_value"}
        )

        # First read should return None (item not found due to eventual consistency)
        result1 = emulator.get_item_by_pk_sk("test_table", "new_item", "test_sk")
        assert result1 is None

        # Second read should return the actual item
        result2 = emulator.get_item_by_pk_sk("test_table", "new_item", "test_sk")
        assert result2["data"] == "new_value"

    def test_eventual_consistency_with_regex_pattern(self):
        """Test eventual consistency simulation with regex pattern matching."""
        config = {
            "enabled": True,
            "delay_reads": 1,
            "patterns": [
                {"table_name": "test_table", "pk_pattern": r"conversation_item_.*"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None, eventual_consistency_config=config
        )

        # Store items with different patterns
        emulator.store_item(
            "test_table",
            {"pk": "conversation_item_123", "sk": "seq_4", "data": "old_value"},
        )
        emulator.store_item(
            "test_table", {"pk": "other_item", "sk": "seq_4", "data": "other_old_value"}
        )

        # Update both items
        emulator.store_item(
            "test_table",
            {"pk": "conversation_item_123", "sk": "seq_4", "data": "new_value"},
        )
        emulator.store_item(
            "test_table", {"pk": "other_item", "sk": "seq_4", "data": "other_new_value"}
        )

        # The conversation_item should have eventual consistency (returns stale data)
        result1 = emulator.get_item_by_pk_sk(
            "test_table", "conversation_item_123", "seq_4"
        )
        assert result1["data"] == "old_value"

        # The other_item should not have eventual consistency (returns new data immediately)
        result2 = emulator.get_item_by_pk_sk("test_table", "other_item", "seq_4")
        assert result2["data"] == "other_new_value"

        # Second read of conversation_item should now return new data
        result3 = emulator.get_item_by_pk_sk(
            "test_table", "conversation_item_123", "seq_4"
        )
        assert result3["data"] == "new_value"

    def test_eventual_consistency_different_delay_counts(self):
        """Test eventual consistency simulation with different delay counts."""
        config = {
            "enabled": True,
            "delay_reads": 3,  # 3 stale reads before returning fresh data
            "patterns": [{"table_name": "test_table", "pk": "test_pk"}],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None, eventual_consistency_config=config
        )

        # Store and update an item
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "old_value"}
        )
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "new_value"}
        )

        # First 3 reads should return stale data
        for i in range(3):
            result = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
            assert result["data"] == "old_value", f"Read {i+1} should return stale data"

        # Fourth read should return fresh data
        result = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result["data"] == "new_value"

    def test_eventual_consistency_only_affects_configured_items(self):
        """Test that eventual consistency only affects items matching the configuration."""
        config = {
            "enabled": True,
            "delay_reads": 1,
            "patterns": [{"table_name": "test_table", "pk": "affected_item"}],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None, eventual_consistency_config=config
        )

        # Store and update two items
        emulator.store_item(
            "test_table", {"pk": "affected_item", "sk": "test_sk", "data": "old_value"}
        )
        emulator.store_item(
            "test_table",
            {"pk": "unaffected_item", "sk": "test_sk", "data": "old_value"},
        )

        emulator.store_item(
            "test_table", {"pk": "affected_item", "sk": "test_sk", "data": "new_value"}
        )
        emulator.store_item(
            "test_table",
            {"pk": "unaffected_item", "sk": "test_sk", "data": "new_value"},
        )

        # Affected item should return stale data
        result1 = emulator.get_item_by_pk_sk("test_table", "affected_item", "test_sk")
        assert result1["data"] == "old_value"

        # Unaffected item should return fresh data immediately
        result2 = emulator.get_item_by_pk_sk("test_table", "unaffected_item", "test_sk")
        assert result2["data"] == "new_value"
