import copy
import threading

from poemai_utils.aws.dynamodb import VersionMismatchException
from sqlitedict import SqliteDict


class DynamoDBEmulator:
    def __init__(self, sqlite_filename):
        self.data_table = SqliteDict(sqlite_filename, tablename="data")
        self.index_table = SqliteDict(sqlite_filename, tablename="index")
        self.lock = threading.Lock()

    def _get_composite_key(self, table_name, pk, sk):
        return f"{table_name}___##___{pk}___##___{sk}"

    def _get_pk_sk_from_composite_key(self, composite_key):
        key_components = composite_key.split("___##___")[1:3]
        return key_components[0], key_components[1]

    def _get_index_key(self, table_name, pk):
        return f"{table_name}#{pk}"

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
            self.data_table.commit()

            index_key = self._get_index_key(table_name, pk)
            index_list = self.index_table.get(index_key, [])

            index_list.append(composite_key)
            self.index_table[index_key] = sorted(index_list)  # Sort the index list
            self.index_table.commit()

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
            self.data_table.commit()

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
        composite_keys = self.index_table.get(index_key, [])
        for composite_key in composite_keys:
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
        self.data_table.commit()

        # Delete the index
        index_key = self._get_index_key(table_name, pk)
        index_list = self.index_table.get(index_key, [])
        index_list.remove(composite_key)
        self.index_table[index_key] = index_list
        self.index_table.commit()

    def scan_for_items_by_pk_sk(self, table_name, pk_contains, sk_contains):
        raise NotImplementedError("scan_for_items_by_pk_sk not implemented")
