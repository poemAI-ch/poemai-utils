import copy
import json
import logging
import re
import threading

from poemai_utils.aws.dynamodb import (
    DynamoDB,
    ItemAlreadyExistsException,
    VersionMismatchException,
)
from sqlitedict import SqliteDict

_logger = logging.getLogger(__name__)


class DynamoDBEmulator:
    def __init__(self, sqlite_filename):
        if sqlite_filename is not None:
            _logger.info(f"Using SQLite data store: {sqlite_filename}")
            self.data_table = SqliteDict(sqlite_filename, tablename="data")
            self.index_table = SqliteDict(sqlite_filename, tablename="index")
            self.is_sqlite = True
        else:
            _logger.info("Using in-memory data store")
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

            # check if the item does not contain unserializeable daata
            assert isinstance(item, dict), f"Item must be a dict, got {type(item)}"
            try:
                _ = json.dumps(item)
            except Exception as e:
                _logger.warning(
                    f"Item {item} is not serializable: {e}, continuing anyway, will probably crash later",
                    exc_info=True,
                )

            serialized_item = DynamoDB.ddb_type_serializer.serialize(item)
            _logger.debug(
                f"Storing serialized_item {serialized_item} with composite key {composite_key}"
            )
            # Store the item
            self.data_table[composite_key] = serialized_item

            index_key = self._get_index_key(table_name, pk)
            index_list = set(self.index_table.get(index_key, []))

            index_list.add(composite_key)

            self.index_table[index_key] = index_list
            self._commit()

    def store_new_item(self, table_name, item, primary_key_name):
        """Store an item only if it does not already exist."""
        pk = item["pk"]
        sk = item.get("sk", "")
        composite_key = self._get_composite_key(table_name, pk, sk)
        if composite_key in self.data_table:
            raise ItemAlreadyExistsException(
                f"Item with pk:{pk} and sk:{sk} already exists."
            )
        self.store_item(table_name, item)

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
            item_serialized = self.data_table.get(composite_key)
            item = DynamoDB.ddb_type_deserializer.deserialize(item_serialized)

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

            serialized_item = DynamoDB.ddb_type_serializer.serialize(item)

            # Store the updated item
            self.data_table[composite_key] = serialized_item
            self._commit()

    def get_item_by_pk_sk(self, table_name, pk, sk):
        composite_key = self._get_composite_key(table_name, pk, sk)

        retval_serialized = self.data_table.get(composite_key, None)
        if retval_serialized is None:
            retval = None
        else:
            retval = DynamoDB.ddb_type_deserializer.deserialize(retval_serialized)

        if retval:
            retval["pk"] = pk
            retval["sk"] = sk
        return retval

    def batch_get_items_by_pk_sk(self, table_name, pk_sk_list):
        _logger.info(
            f"Batch get items by pk_sk list {pk_sk_list} from table {table_name}"
        )
        result_list = []
        for key_spec in pk_sk_list:
            pk = key_spec["pk"]["S"]
            sk = key_spec["sk"]["S"]
            item_found = self.get_item_by_pk_sk(table_name, pk, sk)
            if item_found is not None:
                result_list.append(item_found)
                _logger.info(
                    f"Found item {item_found} for key spec {key_spec}, pk={pk}, sk={sk}"
                )
            else:
                _logger.info(f"Item not found for key spec {key_spec} pk={pk}, sk={sk}")

        return result_list

    def get_item_by_pk(self, table_name, pk):
        composite_key = self._get_composite_key(table_name, pk, "")
        retval_serialized = self.data_table.get(composite_key, None)
        if retval_serialized is None:
            retval = None
        else:
            retval = DynamoDB.ddb_type_deserializer.deserialize(retval_serialized)

        if retval:
            retval["pk"] = pk
        return retval

    def get_paginated_items_by_sk(self, table_name, index_name, sk, limit=100):
        """Get paginated items by sk
        Implmemented as full table scan, very slow, but this is only an emulation anyway....

        Args:
            table_name (str): The name of the table
            index_name (str): The name of the index which has sk as the primary key
            sk (str): The value of the sk
            limit (int): The number of items to return in each page
        """
        for item in self.get_paginated_items(
            table_name=table_name,
            key_condition_expression="sk = :sk",
            expression_attribute_values={":sk": {"S": sk}},
            index_name=index_name,
            limit=limit,
        ):
            yield DynamoDB.item_to_dict(item)

    def get_paginated_items_by_pk(
        self, table_name, pk, limit=None, projection_expression=None
    ):
        results = []
        index_key = self._get_index_key(table_name, pk)
        composite_keys = set(self.index_table.get(index_key, []))
        for composite_key in sorted(composite_keys):
            item_serialized = self.data_table.get(composite_key, None)
            if item_serialized is None:
                item = None
            else:
                item = DynamoDB.ddb_type_deserializer.deserialize(item_serialized)

            if item:
                pk, sk = self._get_pk_sk_from_composite_key(composite_key)
                new_item = copy.deepcopy(item)
                new_item["pk"] = pk
                new_item["sk"] = sk
                if projection_expression:
                    new_item = {
                        k: v
                        for k, v in new_item.items()
                        if k in projection_expression.split(",")
                    }
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
        limit=10000,
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
            if operator == ">=" and item.get(key) >= value:
                return True
            if operator == "<=" and item.get(key) <= value:
                return True
            if operator == ">" and item.get(key) > value:
                return True
            if operator == "<" and item.get(key) < value:
                return True
            if operator == "begins_with" and item.get(key, "").startswith(value):
                return True
            return False

        # Parse the KeyConditionExpression
        conditions = KeyConditionExpression.lower().split(" and ")
        parsed_conditions = []
        for condition in conditions:
            if "begins_with" in condition:
                key, value = re.match(
                    r"begins_with\((\w+), :(\w+)\)", condition
                ).groups()
                operator = "begins_with"
            elif ">=" in condition:
                key, value = re.match(r"(\w+) >= :(\w+)", condition).groups()
                operator = ">="
            elif "<=" in condition:
                key, value = re.match(r"(\w+) <= :(\w+)", condition).groups()
                operator = "<="
            elif ">" in condition:
                key, value = re.match(r"(\w+) > :(\w+)", condition).groups()
                operator = ">"
            elif "<" in condition:
                key, value = re.match(r"(\w+) < :(\w+)", condition).groups()
                operator = "<"
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

        _logger.debug(
            f"Querying table: {TableName}, parsed conditions: {parsed_conditions}"
        )

        # Perform full table scan and filter results
        results = []
        for k, v_serialized in self.data_table.items():
            v = DynamoDB.ddb_type_deserializer.deserialize(v_serialized)

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

        results = sorted(results, key=lambda x: (x.get("pk"), x.get("sk")))

        serialized_results = []
        for item in results:
            _logger.debug(f"Trying to deserialize item {item}")
            for key, value in item.items():
                if hasattr(value, "value"):
                    item[key] = value.value
            serialized_item = DynamoDB.dict_to_item(item)
            serialized_results.append(serialized_item)

        # serialized_results = [DynamoDB.dict_to_item(item) for item in results]

        results = {"Items": serialized_results}

        _logger.debug(f"Query results: {json.dumps(results, indent=2, default=str)}")

        return results

    def item_exists(self, table_name, pk, sk):
        composite_key = self._get_composite_key(table_name, pk, sk)
        return composite_key in self.data_table

    def get_paginated_items(
        self,
        table_name,
        key_condition_expression,
        expression_attribute_values,
        projection_expression=None,
        limit=100,
        index_name=None,
    ):
        # we ignore the index and just do a full table scan
        for i, item in enumerate(
            self.query(
                table_name,
                key_condition_expression,
                expression_attribute_values,
                projection_expression,
                limit=limit,
            )["Items"]
        ):
            if i >= limit:
                break

            # item = {"M": item}
            _logger.debug(f"Yielding item {item}")
            yield item

    def get_paginated_items_starting_at_pk_sk(self, table_name, pk, sk, limit=100):
        """Get paginated items starting at pk, sk, all within the same pk

        Args:
            table_name (str): The name of the table
            pk (str): The value of the pk
            sk (str): The starting value of the sk
            limit (int): The number of items to return in each page
        """
        key_condition_expression = "pk = :pk AND sk >= :sk"
        expression_attribute_values = {
            ":pk": {"S": pk},
            ":sk": {"S": sk},
        }

        for item in self.get_paginated_items(
            table_name=table_name,
            key_condition_expression=key_condition_expression,
            expression_attribute_values=expression_attribute_values,
            limit=limit,
        ):
            item_dict = DynamoDB.item_to_dict(item)
            _logger.debug(f"Yielding item_dict {item_dict}")
            yield item_dict
