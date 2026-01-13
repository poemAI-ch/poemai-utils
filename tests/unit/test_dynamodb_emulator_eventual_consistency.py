"""
Unit tests for the eventual consistency simulator in DynamoDBEmulator.
"""

from poemai_utils.aws.dynamodb_emulator import DynamoDBEmulator


class TestDynamoDBEmulatorEventualConsistency:
    """Test eventual consistency simulation functionality."""

    def test_eventual_consistency_disabled_by_default(self):
        """Test that eventual consistency simulation is disabled by default."""
        emulator = DynamoDBEmulator(
            sqlite_filename=None, allowed_reserved_keywords=["data"]
        )

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
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
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
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
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
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
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

    def test_not_found_reads_override(self):
        """Test that not_found_reads forces initial not-found even when previous versions exist."""
        config = {
            "enabled": True,
            # Separate not-found read plus one stale read
            "delay_reads": 1,
            "not_found_reads": 1,
            "patterns": [
                {"table_name": "test_table", "pk": "test_pk", "sk": "test_sk"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
        )

        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "old_value"}
        )
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "new_value"}
        )

        # First read returns not found even though there is a previous version
        assert emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk") is None
        # Next read still returns stale (old) value because delay_reads=2
        assert (
            emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")["data"]
            == "old_value"
        )
        # Third read returns the fresh value
        assert (
            emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")["data"]
            == "new_value"
        )

    def test_stale_read_counter_counts_not_found_and_stale(self):
        """Verify stale read counter increments for both not-found and stale responses."""
        config = {
            "enabled": True,
            "delay_reads": 1,
            "not_found_reads": 1,
            "patterns": [{"table_name": "test_table", "pk": "count_pk", "sk": "a"}],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
        )

        emulator.store_item("test_table", {"pk": "count_pk", "sk": "a", "data": "old"})
        emulator.store_item("test_table", {"pk": "count_pk", "sk": "a", "data": "new"})

        assert emulator.get_item_by_pk_sk("test_table", "count_pk", "a") is None
        assert (
            emulator.get_item_by_pk_sk("test_table", "count_pk", "a")["data"] == "old"
        )
        assert (
            emulator.get_item_by_pk_sk("test_table", "count_pk", "a")["data"] == "new"
        )

        # One not-found + one stale read
        assert emulator.get_stale_read_count("test_table", "count_pk", "a") == 2

    def test_eventual_consistency_different_delay_counts(self):
        """Test eventual consistency simulation with different delay counts."""
        config = {
            "enabled": True,
            "delay_reads": 3,  # 3 stale reads before returning fresh data
            "patterns": [{"table_name": "test_table", "pk": "test_pk"}],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
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
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
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

    def test_eventual_consistency_delete_operations(self):
        """Test eventual consistency simulation for delete operations."""
        config = {
            "enabled": True,
            "delay_reads": 2,
            "patterns": [
                {"table_name": "test_table", "pk": "test_pk", "sk": "test_sk"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
        )

        # First, store an item without eventual consistency to establish it
        # We'll use a different pattern first, then update to trigger the target pattern
        emulator.store_item(
            "test_table", {"pk": "test_pk_temp", "sk": "test_sk", "data": "test_value"}
        )

        # Verify the temp item exists immediately (no eventual consistency)
        temp_result = emulator.get_item_by_pk_sk(
            "test_table", "test_pk_temp", "test_sk"
        )
        assert temp_result is not None

        # Now store the item that will match our eventual consistency pattern
        # This creates a new item, so it will simulate stale reads by returning None
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "test_value"}
        )

        # Consume the eventual consistency delay from creation (returns None because no previous version)
        for i in range(2):  # delay_reads = 2
            result = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
            assert (
                result is None
            ), f"Read {i+1} should return None due to eventual consistency"

        # Now the item should be consistently readable
        result_consistent = emulator.get_item_by_pk_sk(
            "test_table", "test_pk", "test_sk"
        )
        assert result_consistent is not None
        assert result_consistent["data"] == "test_value"

        # Now delete the item - this should trigger eventual consistency simulation for delete
        emulator.delete_item_by_pk_sk("test_table", "test_pk", "test_sk")

        # First read should still return the deleted item (stale read)
        result1 = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result1 is not None
        assert result1["data"] == "test_value"

        # Second read should still return the deleted item (stale read)
        result2 = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result2 is not None
        assert result2["data"] == "test_value"

        # Third read should return None (item actually deleted)
        result3 = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result3 is None

    def test_eventual_consistency_delete_without_matching_pattern(self):
        """Test that delete operations without matching patterns don't use eventual consistency."""
        config = {
            "enabled": True,
            "delay_reads": 2,
            "patterns": [
                {"table_name": "test_table", "pk": "other_pk"}  # Different pk
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
        )

        # Store an item that doesn't match the pattern
        emulator.store_item(
            "test_table", {"pk": "test_pk", "sk": "test_sk", "data": "test_value"}
        )

        # Verify item exists immediately (no eventual consistency for non-matching pattern)
        result = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result is not None
        assert result["data"] == "test_value"

        # Delete the item - this should NOT trigger eventual consistency
        emulator.delete_item_by_pk_sk("test_table", "test_pk", "test_sk")

        # First read should immediately return None (no eventual consistency)
        result = emulator.get_item_by_pk_sk("test_table", "test_pk", "test_sk")
        assert result is None

    def test_eventual_consistency_delete_with_regex_pattern(self):
        """Test eventual consistency for delete operations with regex pattern matching."""
        config = {
            "enabled": True,
            "delay_reads": 1,
            "patterns": [
                {"table_name": "crawl_requests", "pk_pattern": r"CRAWL_REQUEST#.*"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data", "status"],
        )

        # Store items with matching and non-matching patterns
        emulator.store_item(
            "crawl_requests",
            {"pk": "CRAWL_REQUEST#url1", "sk": "batch1", "status": "processing"},
        )
        emulator.store_item(
            "crawl_requests",
            {"pk": "OTHER_REQUEST#url1", "sk": "batch1", "status": "processing"},
        )

        # Delete both items
        emulator.delete_item_by_pk_sk("crawl_requests", "CRAWL_REQUEST#url1", "batch1")
        emulator.delete_item_by_pk_sk("crawl_requests", "OTHER_REQUEST#url1", "batch1")

        # The CRAWL_REQUEST item should still appear (stale read due to regex match)
        result1 = emulator.get_item_by_pk_sk(
            "crawl_requests", "CRAWL_REQUEST#url1", "batch1"
        )
        assert result1 is not None
        assert result1["status"] == "processing"

        # The OTHER_REQUEST item should be immediately gone (no pattern match)
        result2 = emulator.get_item_by_pk_sk(
            "crawl_requests", "OTHER_REQUEST#url1", "batch1"
        )
        assert result2 is None

        # Second read of CRAWL_REQUEST should now return None
        result3 = emulator.get_item_by_pk_sk(
            "crawl_requests", "CRAWL_REQUEST#url1", "batch1"
        )
        assert result3 is None

    def test_eventual_consistency_delete_crawler_race_condition_simulation(self):
        """Test simulation of the crawler race condition where deleted crawl requests still appear."""
        # This simulates the exact race condition scenario from the crawler
        config = {
            "enabled": True,
            "delay_reads": 2,  # Simulate 2 stale reads before consistency
            "patterns": [
                # Match crawl request items that would be deleted by delete_crawl_request()
                {"table_name": "crawl_requests", "pk_pattern": r"CRAWL_REQUEST#.*"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data", "status"],
        )

        # Set up crawl requests as they would exist in production
        crawl_requests = [
            {
                "pk": "CRAWL_REQUEST#https://example.com/page1",
                "sk": "batch_123",
                "status": "processing",
            },
            {
                "pk": "CRAWL_REQUEST#https://example.com/page2",
                "sk": "batch_123",
                "status": "processing",
            },
            {
                "pk": "CRAWL_REQUEST#https://example.com/page3",
                "sk": "batch_123",
                "status": "completed",
            },
        ]

        # Store all crawl requests
        for request in crawl_requests:
            emulator.store_item("crawl_requests", request)

        # Make items consistently readable by consuming eventual consistency delay from creation
        # Since pattern matches, creation also triggers eventual consistency
        for _ in range(2):  # delay_reads = 2
            for request in crawl_requests:
                _ = emulator.get_item_by_pk_sk(
                    "crawl_requests", request["pk"], request["sk"]
                )

        # Verify all items are now consistently readable
        for request in crawl_requests:
            found = emulator.get_item_by_pk_sk(
                "crawl_requests", request["pk"], request["sk"]
            )
            assert found is not None, f"Request {request['pk']} should be readable"

        # Simulate one Lambda instance processing and deleting requests
        # (this would happen in delete_crawl_request() method)
        for request in crawl_requests:
            emulator.delete_item_by_pk_sk(
                "crawl_requests", request["pk"], request["sk"]
            )

        # Simulate another Lambda instance checking if requests are still open
        # This should see stale data due to eventual consistency, causing the race condition
        remaining_requests = []
        for request in crawl_requests:
            found_request = emulator.get_item_by_pk_sk(
                "crawl_requests", request["pk"], request["sk"]
            )
            if found_request is not None:
                remaining_requests.append(found_request)

        # Due to eventual consistency, we should still see the deleted requests
        assert (
            len(remaining_requests) == 3
        ), f"Expected 3 stale requests, got {len(remaining_requests)}"

        # This simulates the false "Some crawl requests are still open" exception
        # that occurs in production due to this race condition

        # Second read cycle - should still see stale data
        remaining_requests_2nd = []
        for request in crawl_requests:
            found_request = emulator.get_item_by_pk_sk(
                "crawl_requests", request["pk"], request["sk"]
            )
            if found_request is not None:
                remaining_requests_2nd.append(found_request)

        assert (
            len(remaining_requests_2nd) == 3
        ), f"Expected 3 stale requests on 2nd read, got {len(remaining_requests_2nd)}"

        # Third read cycle - should now see consistent (empty) data
        remaining_requests_3rd = []
        for request in crawl_requests:
            found_request = emulator.get_item_by_pk_sk(
                "crawl_requests", request["pk"], request["sk"]
            )
            if found_request is not None:
                remaining_requests_3rd.append(found_request)

        assert (
            len(remaining_requests_3rd) == 0
        ), f"Expected 0 requests on 3rd read (consistent), got {len(remaining_requests_3rd)}"

    def test_eventual_consistency_get_paginated_items_by_pk(self):
        """Test eventual consistency simulation with get_paginated_items_by_pk."""
        config = {
            "enabled": True,
            "delay_reads": 2,
            "patterns": [{"table_name": "test_table", "pk_pattern": r"test_pk"}],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
        )

        # Store some initial items with the same pk
        test_items = [
            {"pk": "test_pk", "sk": "item1", "data": "value1"},
            {"pk": "test_pk", "sk": "item2", "data": "value2"},
            {"pk": "test_pk", "sk": "item3", "data": "value3"},
        ]

        for item in test_items:
            emulator.store_item("test_table", item)

        # Delete all items
        for item in test_items:
            emulator.delete_item_by_pk_sk("test_table", item["pk"], item["sk"])

        # First paginated read after delete - should still see stale data due to eventual consistency
        paginated_results_1st = emulator.get_paginated_items_by_pk(
            "test_table", "test_pk"
        )
        assert (
            len(paginated_results_1st) == 3
        ), f"Expected 3 stale items on 1st paginated read, got {len(paginated_results_1st)}"

        # Second paginated read - should still see stale data
        paginated_results_2nd = emulator.get_paginated_items_by_pk(
            "test_table", "test_pk"
        )
        assert (
            len(paginated_results_2nd) == 3
        ), f"Expected 3 stale items on 2nd paginated read, got {len(paginated_results_2nd)}"

        # Third paginated read - should now see consistent (empty) data
        paginated_results_3rd = emulator.get_paginated_items_by_pk(
            "test_table", "test_pk"
        )
        assert (
            len(paginated_results_3rd) == 0
        ), f"Expected 0 items on 3rd paginated read (consistent), got {len(paginated_results_3rd)}"

    def test_eventual_consistency_get_paginated_items_by_pk_mixed_scenarios(self):
        """Test eventual consistency with get_paginated_items_by_pk in mixed scenarios."""
        config = {
            "enabled": True,
            "delay_reads": 1,
            "patterns": [
                # Only items with sk starting with 'delete_' should have eventual consistency
                {"table_name": "test_table", "pk": "mixed_pk", "sk": "delete_item"}
            ],
        }
        emulator = DynamoDBEmulator(
            sqlite_filename=None,
            eventual_consistency_config=config,
            allowed_reserved_keywords=["data"],
        )

        # Store items - some will have eventual consistency, some won't
        test_items = [
            {"pk": "mixed_pk", "sk": "delete_item", "data": "will_be_stale"},
            {"pk": "mixed_pk", "sk": "keep_item", "data": "immediate_consistency"},
            {"pk": "mixed_pk", "sk": "other_item", "data": "also_immediate"},
        ]

        for item in test_items:
            emulator.store_item("test_table", item)

        # Delete the item that has eventual consistency configured
        emulator.delete_item_by_pk_sk("test_table", "mixed_pk", "delete_item")

        # First paginated read - should see stale data for the deleted item with eventual consistency
        paginated_results = emulator.get_paginated_items_by_pk("test_table", "mixed_pk")

        # Should still see all 3 items (2 regular + 1 stale)
        assert (
            len(paginated_results) == 3
        ), f"Expected 3 items (including stale), got {len(paginated_results)}"

        # Verify the stale item is still present
        stale_items = [
            item for item in paginated_results if item["sk"] == "delete_item"
        ]
        assert len(stale_items) == 1, f"Expected 1 stale item, got {len(stale_items)}"

        # Second paginated read - stale item should now be gone (delay_reads=1)
        paginated_results_2nd = emulator.get_paginated_items_by_pk(
            "test_table", "mixed_pk"
        )
        assert (
            len(paginated_results_2nd) == 2
        ), f"Expected 2 items after consistency, got {len(paginated_results_2nd)}"

        # Verify the stale item is no longer present
        remaining_items = [
            item for item in paginated_results_2nd if item["sk"] == "delete_item"
        ]
        assert (
            len(remaining_items) == 0
        ), f"Expected 0 stale items after consistency, got {len(remaining_items)}"
