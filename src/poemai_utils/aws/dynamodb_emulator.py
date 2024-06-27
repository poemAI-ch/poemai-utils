import copy
import json
import logging
import re
import threading

from poemai_utils.aws.dynamodb import DynamoDB, VersionMismatchException
from sqlitedict import SqliteDict

_logger = logging.getLogger(__name__)


class DynamoDBEmulator:
    def __init__(self, sqlite_filename):
        if sqlite_filename is not None:
            self.data_table = SqliteDict(sqlite_filename, tablename="data")
            self.index_table = SqliteDict(sqlite_filename, tablename="index")
            self.is_sqlite = True
        else:
            self.data_table = {}
            self.index_table = {}
            self.is_sqlite = False
        self.lock = threading.Lock()

    def _get_composite_key(self, table_name, pk, sk):
        return f"{table_name}___##___{pk}___##___{sk}"

    def _get_pk_sk_from_composite_key(self, composite_key):
        key_components = composite_key.split("___##___")[1:3]
        return key_components[0], key_components[1]

    def _get_index_key(self, table_name, pk):
        return f"{table_name}#{pk}"

    def _commit(self):
        if self.is_sqlite:
            self.data_table.commit()
            self.index_table.commit()

    def get_all_items(self):
        for k, v in self.data_table.items():
            pk, sk = self._get_pk_sk_from_composite_key(k)

            yield {"pk": pk, "sk": sk, **v}

    def store_item(self, table_name, item):
        with self.lock:
            pk = item["pk"]
            sk = item.get("sk", "")

            composite_key = self._get_composite_key(table_name, pk, sk)

            # Store the item
            self.data_table[composite_key] = item

            index_key = self._get_index_key(table_name, pk)
            index_list = set(self.index_table.get(index_key, []))

            index_list.add(composite_key)

            self.index_table[index_key] = index_list
            self._commit()

    def update_versioned_item_by_pk_sk(
        self,
        table_name,
        pk,
        sk,
        attribute_updates,
        expected_version,
        version_attribute_name="version",
    ):
        with self.lock:
            composite_key = self._get_composite_key(table_name, pk, sk)
            item = self.data_table.get(composite_key)

            # If the item does not exist, we cannot update it
            if item is None:
                raise KeyError(f"Item with pk:{pk} and sk:{sk} does not exist.")

            # Check for version mismatch
            if item.get(version_attribute_name, 0) != expected_version:
                raise VersionMismatchException(
                    f"Version mismatch for item {pk}:{sk}. "
                    f"Current version: {item.get(version_attribute_name, 0)}, "
                    f"expected: {expected_version}."
                )

            # Update the item's attributes
            for attr, value in attribute_updates.items():
                item[attr] = value

            # Update the version
            item[version_attribute_name] = expected_version + 1

            # Store the updated item
            self.data_table[composite_key] = item
            self._commit()

    def get_item_by_pk_sk(self, table_name, pk, sk):
        composite_key = self._get_composite_key(table_name, pk, sk)

        retval = self.data_table.get(composite_key, None)
        if retval:
            retval["pk"] = pk
            retval["sk"] = sk
        return retval

    def get_item_by_pk(self, table_name, pk):
        composite_key = self._get_composite_key(table_name, pk, "")
        retval = self.data_table.get(composite_key, None)
        if retval:
            retval["pk"] = pk
        return retval

    def get_paginated_items_by_pk(self, table_name, pk, limit=None):
        results = []
        index_key = self._get_index_key(table_name, pk)
        composite_keys = set(self.index_table.get(index_key, []))
        for composite_key in sorted(composite_keys):
            item = self.data_table.get(composite_key, None)
            if item:
                pk, sk = self._get_pk_sk_from_composite_key(composite_key)
                new_item = copy.deepcopy(item)
                new_item["pk"] = pk
                new_item["sk"] = sk
                results.append(new_item)

        return results

    def delete_item_by_pk_sk(self, table_name, pk, sk):
        composite_key = self._get_composite_key(table_name, pk, sk)

        # Delete the item
        del self.data_table[composite_key]

        # Delete the index
        index_key = self._get_index_key(table_name, pk)
        index_list = self.index_table.get(index_key, [])
        index_list.remove(composite_key)
        self.index_table[index_key] = index_list
        self._commit()

    def scan_for_items_by_pk_sk(self, table_name, pk_contains, sk_contains):
        raise NotImplementedError("scan_for_items_by_pk_sk not implemented")

    def query(
        self,
        TableName,
        KeyConditionExpression,
        ExpressionAttributeValues,
        ProjectionExpression=None,
    ):
        """A very simplistic implementation for DynamoDB query operation. It only supports
        equality and begins_with operators in the KeyConditionExpression. It does not
        support any other operations like filter expressions, etc. It also does not
        support any index operations. It is only meant to be used for testing purposes.
        """

        # Helper function to evaluate conditions
        def evaluate_condition(item, key, operator, value):
            if operator == "=" and item.get(key) == value:
                return True
            elif operator == "begins_with" and item.get(key, "").startswith(value):
                return True
            return False

        # Parse the KeyConditionExpression
        conditions = KeyConditionExpression.split(" and ")
        parsed_conditions = []
        for condition in conditions:
            if "begins_with" in condition:
                key, value = re.match(
                    r"begins_with\((\w+), :(\w+)\)", condition
                ).groups()
                operator = "begins_with"
            else:
                key, value = re.match(r"(\w+) = :(\w+)", condition).groups()
                operator = "="
            parsed_conditions.append((key, operator, value))

        # Replace placeholders with actual values
        for i, (key, operator, placeholder) in enumerate(parsed_conditions):
            value_dict = ExpressionAttributeValues.get(f":{placeholder}")
            if value_dict:
                value = next(
                    iter(value_dict.values())
                )  # Get the value from dict e.g., {"S": "some_value"}
                parsed_conditions[i] = (key, operator, value)

        # Perform full table scan and filter results
        results = []
        for k, v in self.data_table.items():
            # Extract table name, pk, and sk from the composite key
            key_parts = k.split("___##___")
            if key_parts[0] != TableName:
                continue  # Skip items that do not belong to the specified table

            pk, sk = key_parts[1], key_parts[2]
            item = {"pk": pk, "sk": sk, **v}
            # Check all conditions
            if all(
                evaluate_condition(item, key, operator, value)
                for key, operator, value in parsed_conditions
            ):
                # If projection is specified, filter the keys
                if ProjectionExpression:
                    projected_item = {
                        k: v
                        for k, v in item.items()
                        if k in ProjectionExpression.split(",")
                    }
                    results.append(projected_item)
                else:
                    results.append(item)

        results = {"Items": [DynamoDB.dict_to_item(item) for item in results]}

        _logger.info(f"Query results: {json.dumps(results, indent=2)}")

        return results

    def item_exists(self, table_name, pk, sk):
        composite_key = self._get_composite_key(table_name, pk, sk)
        return composite_key in self.data_table
