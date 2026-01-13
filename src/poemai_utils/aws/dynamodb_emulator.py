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


def _make_json_serializable_for_testing(obj):
    """
    Recursively converts objects to JSON-serializable forms for DynamoDB emulator testing.
    Handles DynamoDB-supported data types like Decimal and binary data.

    This is intended for validation/testing, not for actual data storage.
    """
    import base64
    from decimal import Decimal

    if isinstance(obj, dict):
        return {
            key: _make_json_serializable_for_testing(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [_make_json_serializable_for_testing(element) for element in obj]
    elif isinstance(obj, Decimal):
        return str(obj)  # Convert Decimal to string
    elif isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("ascii")  # Convert binary to base64 string
    else:
        return obj


# DynamoDB reserved keywords (case-insensitive)
# Source: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
DYNAMODB_RESERVED_KEYWORDS = {
    "ABORT",
    "ABSOLUTE",
    "ACTION",
    "ADD",
    "AFTER",
    "AGENT",
    "AGGREGATE",
    "ALL",
    "ALLOCATE",
    "ALTER",
    "ANALYZE",
    "AND",
    "ANY",
    "ARCHIVE",
    "ARE",
    "ARRAY",
    "AS",
    "ASC",
    "ASCII",
    "ASENSITIVE",
    "ASSERTION",
    "ASYMMETRIC",
    "AT",
    "ATOMIC",
    "ATTACH",
    "ATTRIBUTE",
    "AUTH",
    "AUTHORIZATION",
    "AUTHORIZE",
    "AUTO",
    "AVG",
    "BACK",
    "BACKUP",
    "BASE",
    "BATCH",
    "BEFORE",
    "BEGIN",
    "BETWEEN",
    "BIGINT",
    "BINARY",
    "BIT",
    "BLOB",
    "BLOCK",
    "BOOLEAN",
    "BOTH",
    "BREADTH",
    "BUCKET",
    "BULK",
    "BY",
    "BYTE",
    "CALL",
    "CALLED",
    "CALLING",
    "CAPACITY",
    "CASCADE",
    "CASCADED",
    "CASE",
    "CAST",
    "CATALOG",
    "CHAR",
    "CHARACTER",
    "CHECK",
    "CLASS",
    "CLOB",
    "CLOSE",
    "CLUSTER",
    "CLUSTERED",
    "CLUSTERING",
    "CLUSTERS",
    "COALESCE",
    "COLLATE",
    "COLLATION",
    "COLLECTION",
    "COLUMN",
    "COLUMNS",
    "COMBINE",
    "COMMENT",
    "COMMIT",
    "COMPACT",
    "COMPILE",
    "COMPRESS",
    "CONDITION",
    "CONFLICT",
    "CONNECT",
    "CONNECTION",
    "CONSISTENCY",
    "CONSISTENT",
    "CONSTRAINT",
    "CONSTRAINTS",
    "CONSTRUCTOR",
    "CONSUMED",
    "CONTINUE",
    "CONVERT",
    "COPY",
    "CORRESPONDING",
    "COUNT",
    "COUNTER",
    "CREATE",
    "CROSS",
    "CUBE",
    "CURRENT",
    "CURSOR",
    "CYCLE",
    "DATA",
    "DATABASE",
    "DATE",
    "DATETIME",
    "DAY",
    "DEALLOCATE",
    "DEC",
    "DECIMAL",
    "DECLARE",
    "DEFAULT",
    "DEFERRABLE",
    "DEFERRED",
    "DEFINE",
    "DEFINED",
    "DEFINITION",
    "DELETE",
    "DELIMITED",
    "DEPTH",
    "DEREF",
    "DESC",
    "DESCRIBE",
    "DESCRIPTOR",
    "DETACH",
    "DETERMINISTIC",
    "DIAGNOSTICS",
    "DIRECTORIES",
    "DISABLE",
    "DISCONNECT",
    "DISTINCT",
    "DISTRIBUTE",
    "DO",
    "DOMAIN",
    "DOUBLE",
    "DROP",
    "DUMP",
    "DURATION",
    "DYNAMIC",
    "EACH",
    "ELEMENT",
    "ELSE",
    "ELSEIF",
    "EMPTY",
    "ENABLE",
    "END",
    "EQUAL",
    "EQUALS",
    "ERROR",
    "ESCAPE",
    "ESCAPED",
    "EVAL",
    "EVALUATE",
    "EXCEEDED",
    "EXCEPT",
    "EXCEPTION",
    "EXCEPTIONS",
    "EXCLUSIVE",
    "EXEC",
    "EXECUTE",
    "EXISTS",
    "EXIT",
    "EXPLAIN",
    "EXPLODE",
    "EXPORT",
    "EXPRESSION",
    "EXTENDED",
    "EXTERNAL",
    "EXTRACT",
    "FAIL",
    "FALSE",
    "FAMILY",
    "FETCH",
    "FIELDS",
    "FILE",
    "FILTER",
    "FILTERING",
    "FINAL",
    "FINISH",
    "FIRST",
    "FIXED",
    "FLATTERN",
    "FLOAT",
    "FOR",
    "FORCE",
    "FOREIGN",
    "FORMAT",
    "FORWARD",
    "FOUND",
    "FREE",
    "FROM",
    "FULL",
    "FUNCTION",
    "FUNCTIONS",
    "GENERAL",
    "GENERATE",
    "GET",
    "GLOB",
    "GLOBAL",
    "GO",
    "GOTO",
    "GRANT",
    "GREATER",
    "GROUP",
    "GROUPING",
    "HANDLER",
    "HASH",
    "HAVE",
    "HAVING",
    "HEAP",
    "HIDDEN",
    "HOLD",
    "HOUR",
    "IDENTIFIED",
    "IDENTITY",
    "IF",
    "IGNORE",
    "IMMEDIATE",
    "IMPORT",
    "IN",
    "INCLUDING",
    "INCLUSIVE",
    "INCREMENT",
    "INCREMENTAL",
    "INDEX",
    "INDEXED",
    "INDEXES",
    "INDICATOR",
    "INFINITE",
    "INITIALLY",
    "INLINE",
    "INNER",
    "INNTER",
    "INOUT",
    "INPUT",
    "INSENSITIVE",
    "INSERT",
    "INSTEAD",
    "INT",
    "INTEGER",
    "INTERSECT",
    "INTERVAL",
    "INTO",
    "INVALIDATE",
    "IS",
    "ISOLATION",
    "ITEM",
    "ITEMS",
    "ITERATE",
    "JOIN",
    "KEY",
    "KEYS",
    "LAG",
    "LANGUAGE",
    "LARGE",
    "LAST",
    "LATERAL",
    "LEAD",
    "LEADING",
    "LEAVE",
    "LEFT",
    "LENGTH",
    "LESS",
    "LEVEL",
    "LIKE",
    "LIMIT",
    "LIMITED",
    "LINES",
    "LIST",
    "LOAD",
    "LOCAL",
    "LOCALTIME",
    "LOCALTIMESTAMP",
    "LOCATION",
    "LOCATOR",
    "LOCK",
    "LOCKS",
    "LOG",
    "LOGED",
    "LONG",
    "LOOP",
    "LOWER",
    "MAP",
    "MATCH",
    "MATERIALIZED",
    "MAX",
    "MAXLEN",
    "MEMBER",
    "MERGE",
    "METHOD",
    "METRICS",
    "MIN",
    "MINUS",
    "MINUTE",
    "MISSING",
    "MOD",
    "MODE",
    "MODIFIES",
    "MODIFY",
    "MODULE",
    "MONTH",
    "MULTI",
    "MULTISET",
    "NAME",
    "NAMES",
    "NATIONAL",
    "NATURAL",
    "NCHAR",
    "NCLOB",
    "NEW",
    "NEXT",
    "NO",
    "NONE",
    "NOT",
    "NULL",
    "NULLIF",
    "NUMBER",
    "NUMERIC",
    "OBJECT",
    "OF",
    "OFFLINE",
    "OFFSET",
    "OLD",
    "ON",
    "ONLINE",
    "ONLY",
    "OPAQUE",
    "OPEN",
    "OPERATOR",
    "OPTION",
    "OR",
    "ORDER",
    "ORDINALITY",
    "OTHER",
    "OTHERS",
    "OUT",
    "OUTER",
    "OUTPUT",
    "OVER",
    "OVERLAPS",
    "OVERRIDE",
    "OWNER",
    "PAD",
    "PARALLEL",
    "PARAMETER",
    "PARAMETERS",
    "PARTIAL",
    "PARTITION",
    "PARTITIONED",
    "PARTITIONS",
    "PATH",
    "PERCENT",
    "PERCENTILE",
    "PERMISSION",
    "PERMISSIONS",
    "PIPE",
    "PIPELINED",
    "PLAN",
    "POOL",
    "POSITION",
    "PRECISION",
    "PREPARE",
    "PRESERVE",
    "PRIMARY",
    "PRIOR",
    "PRIVATE",
    "PRIVILEGES",
    "PROCEDURE",
    "PROCESSED",
    "PROJECT",
    "PROJECTION",
    "PROPERTY",
    "PROVISIONING",
    "PUBLIC",
    "PUT",
    "QUERY",
    "QUIT",
    "QUORUM",
    "RAISE",
    "RANDOM",
    "RANGE",
    "RANK",
    "RAW",
    "READ",
    "READS",
    "REAL",
    "REBUILD",
    "RECORD",
    "RECURSIVE",
    "REDUCE",
    "REF",
    "REFERENCE",
    "REFERENCES",
    "REFERENCING",
    "REGEXP",
    "REGION",
    "REINDEX",
    "RELATIVE",
    "RELEASE",
    "REMAINDER",
    "RENAME",
    "REPEAT",
    "REPLACE",
    "REQUEST",
    "RESET",
    "RESIGNAL",
    "RESOURCE",
    "RESPONSE",
    "RESTORE",
    "RESTRICT",
    "RESULT",
    "RETURN",
    "RETURNING",
    "RETURNS",
    "REVERSE",
    "REVOKE",
    "RIGHT",
    "ROLE",
    "ROLES",
    "ROLLBACK",
    "ROLLUP",
    "ROUTINE",
    "ROW",
    "ROWS",
    "RULE",
    "RULES",
    "SAMPLE",
    "SATISFIES",
    "SAVE",
    "SAVEPOINT",
    "SCAN",
    "SCHEMA",
    "SCOPE",
    "SCROLL",
    "SEARCH",
    "SECOND",
    "SECTION",
    "SEGMENT",
    "SEGMENTS",
    "SELECT",
    "SELF",
    "SEMI",
    "SENSITIVE",
    "SEPARATE",
    "SEQUENCE",
    "SERIALIZABLE",
    "SESSION",
    "SET",
    "SETS",
    "SHARD",
    "SHARE",
    "SHARED",
    "SHORT",
    "SHOW",
    "SIGNAL",
    "SIMILAR",
    "SIZE",
    "SKEWED",
    "SMALLINT",
    "SNAPSHOT",
    "SOME",
    "SOURCE",
    "SPACE",
    "SPACES",
    "SPARSE",
    "SPECIFIC",
    "SPECIFICTYPE",
    "SPLIT",
    "SQL",
    "SQLCODE",
    "SQLERROR",
    "SQLEXCEPTION",
    "SQLSTATE",
    "SQLWARNING",
    "START",
    "STATE",
    "STATIC",
    "STATUS",
    "STORAGE",
    "STORE",
    "STORED",
    "STREAM",
    "STRING",
    "STRUCT",
    "STYLE",
    "SUB",
    "SUBMULTISET",
    "SUBPARTITION",
    "SUBSTRING",
    "SUBTYPE",
    "SUM",
    "SUPER",
    "SYMMETRIC",
    "SYNONYM",
    "SYSTEM",
    "TABLE",
    "TABLESAMPLE",
    "TEMP",
    "TEMPORARY",
    "TERMINATED",
    "TEXT",
    "THAN",
    "THEN",
    "THROUGHPUT",
    "TIME",
    "TIMESTAMP",
    "TIMEZONE",
    "TINYINT",
    "TO",
    "TOKEN",
    "TOTAL",
    "TOUCH",
    "TRAILING",
    "TRANSACTION",
    "TRANSFORM",
    "TRANSLATE",
    "TRANSLATION",
    "TREAT",
    "TRIGGER",
    "TRIM",
    "TRUE",
    "TRUNCATE",
    "TTL",
    "TUPLE",
    "TYPE",
    "UNDER",
    "UNDO",
    "UNION",
    "UNIQUE",
    "UNIT",
    "UNKNOWN",
    "UNLOGGED",
    "UNNEST",
    "UNPROCESSED",
    "UNSIGNED",
    "UNTIL",
    "UPDATE",
    "UPPER",
    "URL",
    "USAGE",
    "USE",
    "USER",
    "USERS",
    "USING",
    "UUID",
    "VACUUM",
    "VALUE",
    "VALUED",
    "VALUES",
    "VARCHAR",
    "VARIABLE",
    "VARIANCE",
    "VARINT",
    "VARYING",
    "VIEW",
    "VIEWS",
    "VIRTUAL",
    "VOID",
    "WAIT",
    "WHEN",
    "WHENEVER",
    "WHERE",
    "WHILE",
    "WINDOW",
    "WITH",
    "WITHIN",
    "WITHOUT",
    "WORK",
    "WRAPPED",
    "WRITE",
    "YEAR",
    "ZONE",
}


def validate_attribute_names(data, context="operation", allowed_keywords=None):
    """
    Validate that attribute names do not contain DynamoDB reserved keywords.

    Args:
        data: Can be a dict (item), list of strings (attribute names), or string (single attribute)
        context: String describing the context for error messages
        allowed_keywords: Set of reserved keywords that are allowed (case-insensitive)

    Raises:
        Exception: If any reserved keywords are found
    """
    allowed_keywords = set(keyword.upper() for keyword in (allowed_keywords or []))
    reserved_found = []

    if isinstance(data, dict):
        # Check all keys in a dictionary
        for key in data.keys():
            if (
                key.upper() in DYNAMODB_RESERVED_KEYWORDS
                and key.upper() not in allowed_keywords
            ):
                reserved_found.append(key)
    elif isinstance(data, list):
        # Check all strings in a list
        for attr_name in data:
            if (
                attr_name.upper() in DYNAMODB_RESERVED_KEYWORDS
                and attr_name.upper() not in allowed_keywords
            ):
                reserved_found.append(attr_name)
    elif isinstance(data, str):
        # Check a single string
        if (
            data.upper() in DYNAMODB_RESERVED_KEYWORDS
            and data.upper() not in allowed_keywords
        ):
            reserved_found.append(data)

    if reserved_found:
        raise Exception(
            f"DynamoDB {context} contains reserved keyword(s): {reserved_found}. "
            f"Use expression attribute names to work with reserved keywords. Pass allowed_reserved_keywords to the constructor to allow keywords explicitly, for testing. "
            f"See: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ExpressionAttributeNames.html"
        )


class DynamoDBEmulator:
    def __init__(
        self,
        sqlite_filename,
        enforce_index_existence=False,
        eventual_consistency_config=None,
        log_access=False,
        allowed_reserved_keywords=None,
    ):
        self.log_access = log_access
        self.allowed_reserved_keywords = set(
            keyword.upper() for keyword in (allowed_reserved_keywords or [])
        )
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
        self.enforce_index_existence = enforce_index_existence
        self.indexes = (
            {}
        )  # Store index definitions: {table_name: {index_name: index_spec}}

        # Eventual consistency simulation
        self.eventual_consistency_config = eventual_consistency_config or {}
        self.stale_reads_remaining = (
            {}
        )  # tracks how many stale reads remain for each key
        self.not_found_reads_remaining = (
            {}
        )  # tracks how many reads should return not found for each key
        self.stale_reads_served_count = (
            {}
        )  # tracks how many stale/not-found reads were served per key
        self.previous_versions = {}  # stores previous versions of items
        self.shadow_index = (
            {}
        )  # tracks deleted items still visible due to eventual consistency
        self.key_schema_by_table = {}

    def _get_composite_key(self, table_name, pk, sk):
        return f"{table_name}___##___{pk}___##___{sk}"

    def _get_pk_sk_from_composite_key(self, composite_key):
        key_components = composite_key.split("___##___")[1:3]
        return key_components[0], key_components[1]

    def _get_table_name_from_composite_key(self, composite_key):
        return composite_key.split("___##___")[0]

    def _get_index_key(self, table_name, pk):
        return f"{table_name}#{pk}"

    def _commit(self):
        if self.is_sqlite:
            self.data_table.commit()
            self.index_table.commit()

    def _get_key_names(self, table_name):
        """Get the primary key and sort key attribute names for a table.

        Args:
            table_name (str): The name of the table

        Returns:
            tuple: (pk_key, sk_key) where pk_key is the primary key attribute name
                   and sk_key is the sort key attribute name (or None if no sort key)
        """
        pk_key = "pk"
        sk_key = "sk"
        if table_name in self.key_schema_by_table:
            sk_key = None
            for key_def in self.key_schema_by_table[table_name]:
                if key_def["KeyType"] == "HASH":
                    pk_key = key_def["AttributeName"]
                elif key_def["KeyType"] == "RANGE":
                    sk_key = key_def["AttributeName"]
        return pk_key, sk_key

    def add_key_schema(self, table_name, key_schema):
        """Add key schema for a table. Key schema is a list of dicts with 'AttributeName' and 'KeyType'."""
        if self.log_access:
            _logger.info(f"Adding key schema for table {table_name}: {key_schema}")
        self.key_schema_by_table[table_name] = key_schema

    def get_all_items(self):
        if self.log_access:
            _logger.info("Getting all items from all tables")
        for k, v in self.data_table.items():
            _logger.info(f"Got item with composite key {k}")
            pk, sk = self._get_pk_sk_from_composite_key(k)
            table_name = self._get_table_name_from_composite_key(k)

            pk_key, sk_key = self._get_key_names(table_name)

            retval = {pk_key: pk}
            if sk_key:
                retval[sk_key] = sk
            yield {**retval, **v}

    def store_item(self, table_name, item):
        if self.log_access:
            _logger.info(f"Storing item in table {table_name}: {item}")

        # Validate attribute names against DynamoDB reserved keywords
        # Allow certain keywords for item storage but not for expressions
        validate_attribute_names(
            item,
            f"put_item for table '{table_name}'",
            allowed_keywords=self.allowed_reserved_keywords,
        )

        with self.lock:
            pk_key, sk_key = self._get_key_names(table_name)

            pk = item[pk_key]
            sk = item.get(sk_key, "")

            composite_key = self._get_composite_key(table_name, pk, sk)

            # check if the item does not contain unserializeable data
            assert isinstance(item, dict), f"Item must be a dict, got {type(item)}"
            try:
                # Convert Decimals and binary data to JSON-serializable forms for testing
                serializable_item = _make_json_serializable_for_testing(item)
                _ = json.dumps(serializable_item)
            except Exception as e:
                _logger.warning(
                    f"Item {str(item)[:300]} is not serializable: {e}",
                    exc_info=True,
                )
                raise TypeError(
                    f"Item contains unserializable data (e.g. MagicMock, arbitrary objects, ...). Error: {e}"
                ) from e

            # Store previous version for eventual consistency simulation
            if composite_key in self.data_table:
                self.previous_versions[composite_key] = self.data_table[composite_key]

            # Check if this item should have eventual consistency simulation
            pattern_match = self._match_eventual_pattern(table_name, pk, sk)
            if pattern_match:
                delay_reads = pattern_match.get(
                    "delay_reads",
                    self.eventual_consistency_config.get("delay_reads", 2),
                )
                self.stale_reads_remaining[composite_key] = delay_reads
                not_found_reads = pattern_match.get(
                    "not_found_reads",
                    self.eventual_consistency_config.get("not_found_reads", 0),
                )
                if not_found_reads > 0:
                    self.not_found_reads_remaining[composite_key] = not_found_reads
                _logger.debug(
                    f"Configured {delay_reads} stale reads and {not_found_reads} not-found reads for {composite_key}"
                )

            serialized_item = DynamoDB.ddb_type_serializer.serialize(item)
            _logger.debug(f"Storing serialized_item with composite key {composite_key}")
            # Store the item
            self.data_table[composite_key] = serialized_item

            index_key = self._get_index_key(table_name, pk)
            index_list = set(self.index_table.get(index_key, []))

            index_list.add(composite_key)

            self.index_table[index_key] = index_list
            self._commit()

    def _match_eventual_pattern(self, table_name, pk, sk):
        """
        Determine if eventual consistency should be simulated for this item and return the matched pattern.
        """
        if not self.eventual_consistency_config.get("enabled", False):
            return None

        patterns = self.eventual_consistency_config.get("patterns", [])
        for pattern in patterns:
            if pattern.get("table_name") == table_name:
                if "pk_pattern" in pattern:
                    if re.match(pattern["pk_pattern"], pk):
                        return pattern
                elif "pk" in pattern and pattern["pk"] == pk:
                    if "sk" in pattern and pattern["sk"] == sk:
                        return pattern
                    elif "sk" not in pattern:
                        return pattern
        return None

    def store_new_item(self, table_name, item, primary_key_name):
        """Store an item only if it does not already exist."""
        if self.log_access:
            _logger.info(
                f"Storing new item in table {table_name} with primary key {primary_key_name}: {item}"
            )
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

        if self.log_access:
            _logger.info(
                f"Updating item in table {table_name} with pk:{pk}, sk:{sk}, expected_version:{expected_version}, updates:{attribute_updates}"
            )

        # Validate attribute names against DynamoDB reserved keywords
        # Allow certain keywords for item updates but not for expressions
        validate_attribute_names(
            attribute_updates,
            f"update_item for table '{table_name}'",
            allowed_keywords=self.allowed_reserved_keywords,
        )

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

    def get_item(
        self, TableName, Key, ProjectionExpression=None, ExpressionAttributeNames=None
    ):

        pk_key, sk_key = self._get_key_names(TableName)

        pk = Key[pk_key]["S"]

        if sk_key and sk_key in Key:
            sk = Key[sk_key]["S"]
        else:
            sk = None

        if sk is None:
            _logger.debug(f"Getting item from table {TableName} by pk only: {pk}")
            raw_item = self.get_item_by_pk(TableName, pk)
        else:
            _logger.debug(
                f"Getting item from table {TableName} by pk and sk: {pk}, {sk}"
            )
            raw_item = self.get_item_by_pk_sk(TableName, pk, sk)
        if raw_item is None:
            return None

        if ProjectionExpression is not None and raw_item is not None:
            projected_keys = [
                k.strip() for k in ProjectionExpression.split(",") if k.strip()
            ]

            # Resolve ExpressionAttributeNames if provided
            resolved_keys = []
            if ExpressionAttributeNames:
                for key in projected_keys:
                    if key.startswith("#"):
                        # This is an expression attribute name
                        if key in ExpressionAttributeNames:
                            resolved_keys.append(ExpressionAttributeNames[key])
                        else:
                            raise Exception(
                                f"ExpressionAttributeName '{key}' not found in ExpressionAttributeNames"
                            )
                    else:
                        resolved_keys.append(key)
                # When ExpressionAttributeNames are used, validate the original expression keys (which should be aliases)
                validate_attribute_names(
                    projected_keys, f"ProjectionExpression for table '{TableName}'"
                )
            else:
                resolved_keys = projected_keys
                # When no ExpressionAttributeNames, validate the actual attribute names
                validate_attribute_names(
                    resolved_keys, f"ProjectionExpression for table '{TableName}'"
                )

            projected_item = {k: v for k, v in raw_item.items() if k in resolved_keys}
        else:
            projected_item = raw_item

        retval = {"Item": DynamoDB.ddb_type_serializer.serialize(projected_item)["M"]}
        return retval

    def get_item_by_pk_sk(self, table_name, pk, sk):
        if self.log_access:
            _logger.info(f"Getting item from table {table_name} with pk:{pk}, sk:{sk}")
        composite_key = self._get_composite_key(table_name, pk, sk)

        pk_key, sk_key = self._get_key_names(table_name)

        # Check if this read should return stale data due to eventual consistency
        with self.lock:
            if composite_key in self.not_found_reads_remaining:
                if self.not_found_reads_remaining[composite_key] > 0:
                    self.not_found_reads_remaining[composite_key] -= 1
                    self.stale_reads_served_count[composite_key] = (
                        self.stale_reads_served_count.get(composite_key, 0) + 1
                    )
                    _logger.info(
                        f"Returning simulated not-found for {composite_key}, {self.not_found_reads_remaining[composite_key]} not-found reads remaining"
                    )
                    if self.not_found_reads_remaining[composite_key] == 0:
                        del self.not_found_reads_remaining[composite_key]
                    if (
                        composite_key in self.stale_reads_remaining
                        and self.stale_reads_remaining[composite_key] == 0
                    ):
                        self.previous_versions.pop(composite_key, None)
                        self.stale_reads_remaining.pop(composite_key, None)
                    return None
            if (
                composite_key in self.stale_reads_remaining
                and self.stale_reads_remaining[composite_key] > 0
            ):
                self.stale_reads_remaining[composite_key] -= 1
                self.stale_reads_served_count[composite_key] = (
                    self.stale_reads_served_count.get(composite_key, 0) + 1
                )
                stale_data = self.previous_versions.get(composite_key)
                _logger.info(
                    f"Returning stale data for {composite_key}, {self.stale_reads_remaining[composite_key]} stale reads remaining"
                )

                if self.stale_reads_remaining[composite_key] == 0:
                    del self.stale_reads_remaining[composite_key]
                    if composite_key in self.previous_versions:
                        del self.previous_versions[composite_key]

                if stale_data is not None:
                    retval = DynamoDB.ddb_type_deserializer.deserialize(stale_data)
                    if retval:
                        retval[pk_key] = pk
                        retval[sk_key] = sk
                    return retval
                else:
                    # No previous version exists, return None to simulate item not found
                    return None

        # Return current data
        retval_serialized = self.data_table.get(composite_key, None)
        if retval_serialized is None:
            retval = None
        else:
            retval = DynamoDB.ddb_type_deserializer.deserialize(retval_serialized)

        if retval:
            retval[pk_key] = pk
            retval[sk_key] = sk
        return retval

    def batch_get_items_by_pk_sk(self, table_name, pk_sk_list):
        if self.log_access:
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
                if self.log_access:
                    _logger.info(
                        f"Found item {item_found} for key spec {key_spec}, pk={pk}, sk={sk}"
                    )
            else:
                if self.log_access:
                    _logger.info(
                        f"Item not found for key spec {key_spec} pk={pk}, sk={sk}"
                    )

        return result_list

    def get_item_by_pk(self, table_name, pk):
        composite_key = self._get_composite_key(table_name, pk, "")
        pk_key, sk_key = self._get_key_names(table_name)

        # Validate that this table doesn't have a sort key
        if sk_key is not None:
            raise ValueError(
                f"Table {table_name} has a sort key, cannot get item by pk only"
            )

        if self.log_access:
            _logger.info(
                f"Getting item from table {table_name} with pk:{pk}, composite_key:{composite_key}"
            )

        retval_serialized = self.data_table.get(composite_key, None)

        if retval_serialized is None:
            _logger.info(f"retval_serialized is None, returning None")
            retval = None
        else:
            retval = DynamoDB.ddb_type_deserializer.deserialize(retval_serialized)

        if retval:
            retval[pk_key] = pk

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
        if self.log_access:
            _logger.info(
                f"Getting get_paginated_items_by_sk sk:{sk} from table {table_name} using index {index_name} with limit {limit}"
            )
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
        if self.log_access:
            _logger.info(
                f"get_paginated_items_by_pk pk:{pk} from table {table_name} with limit {limit} and projection_expression {projection_expression}"
            )
        results = []
        index_key = self._get_index_key(table_name, pk)

        # Get composite keys from both regular index and shadow index (for eventual consistency)
        regular_keys = set(self.index_table.get(index_key, []))
        shadow_keys = set(self.shadow_index.get(index_key, []))
        all_composite_keys = regular_keys.union(shadow_keys)

        pk_key, sk_key = self._get_key_names(table_name)

        # Track keys to remove from shadow index after eventual consistency expires
        shadow_keys_to_remove = []

        for composite_key in sorted(all_composite_keys):
            current_pk, current_sk = self._get_pk_sk_from_composite_key(composite_key)

            # Check if this read should return stale data due to eventual consistency
            with self.lock:
                if (
                    composite_key in self.not_found_reads_remaining
                    and self.not_found_reads_remaining[composite_key] > 0
                ):
                    self.not_found_reads_remaining[composite_key] -= 1
                    self.stale_reads_served_count[composite_key] = (
                        self.stale_reads_served_count.get(composite_key, 0) + 1
                    )
                    _logger.debug(
                        f"Returning simulated not-found for {composite_key} in paginated query, {self.not_found_reads_remaining[composite_key]} not-found reads remaining"
                    )
                    if self.not_found_reads_remaining[composite_key] == 0:
                        del self.not_found_reads_remaining[composite_key]
                    shadow_keys_to_remove.append(composite_key)
                    continue
                if (
                    composite_key in self.stale_reads_remaining
                    and self.stale_reads_remaining[composite_key] > 0
                ):
                    self.stale_reads_remaining[composite_key] -= 1
                    self.stale_reads_served_count[composite_key] = (
                        self.stale_reads_served_count.get(composite_key, 0) + 1
                    )
                    stale_data = self.previous_versions.get(composite_key)
                    _logger.debug(
                        f"Returning stale data for {composite_key} in paginated query, {self.stale_reads_remaining[composite_key]} stale reads remaining"
                    )

                    # If this was the last stale read, mark for shadow index cleanup
                    if self.stale_reads_remaining[composite_key] == 0:
                        shadow_keys_to_remove.append(composite_key)
                        # Clean up stale reads tracking
                        del self.stale_reads_remaining[composite_key]
                        if composite_key in self.previous_versions:
                            del self.previous_versions[composite_key]

                    if stale_data is not None:
                        item = DynamoDB.ddb_type_deserializer.deserialize(stale_data)
                        if item:
                            new_item = copy.deepcopy(item)
                            new_item[pk_key] = current_pk
                            new_item[sk_key] = current_sk
                            if projection_expression:
                                new_item = {
                                    k: v
                                    for k, v in new_item.items()
                                    if k in projection_expression.split(",")
                                }
                            results.append(new_item)
                    # If stale_data is None, skip this item (simulates item not found due to eventual consistency)
                    continue

            # Return current data if no stale read simulation
            item_serialized = self.data_table.get(composite_key, None)
            if item_serialized is not None:
                # Item exists in data table - return it normally
                item = DynamoDB.ddb_type_deserializer.deserialize(item_serialized)
                if item:
                    new_item = copy.deepcopy(item)
                    new_item[pk_key] = current_pk
                    new_item[sk_key] = current_sk
                    if projection_expression:
                        new_item = {
                            k: v
                            for k, v in new_item.items()
                            if k in projection_expression.split(",")
                        }
                    results.append(new_item)
            # If item doesn't exist in data table but is in shadow index,
            # it means it's been deleted and eventual consistency has expired - skip it

        # Clean up shadow index entries for items that no longer need to be tracked
        if shadow_keys_to_remove:
            with self.lock:
                shadow_set = self.shadow_index.get(index_key, set())
                for key_to_remove in shadow_keys_to_remove:
                    if key_to_remove in shadow_set:
                        shadow_set.remove(key_to_remove)
                        _logger.debug(
                            f"Removed {key_to_remove} from shadow index after eventual consistency expired"
                        )

                # Clean up empty shadow index entries
                if not shadow_set:
                    if index_key in self.shadow_index:
                        del self.shadow_index[index_key]
                else:
                    self.shadow_index[index_key] = shadow_set

        return results

    def delete_item_by_pk_sk(self, table_name, pk, sk):
        if self.log_access:
            _logger.info(f"Deleting item from table {table_name} with pk:{pk}, sk:{sk}")
        composite_key = self._get_composite_key(table_name, pk, sk)

        with self.lock:
            # Store the item as a previous version for eventual consistency simulation
            if composite_key in self.data_table:
                self.previous_versions[composite_key] = self.data_table[composite_key]

            # Check if this delete should have eventual consistency simulation
            pattern_match = self._match_eventual_pattern(table_name, pk, sk)
            if pattern_match:
                _logger.debug(
                    f"Simulating eventual consistency for delete operation on {pk}:{sk}"
                )
                # Configure how many stale reads should still return the deleted item
                delay_reads = pattern_match.get(
                    "delay_reads",
                    self.eventual_consistency_config.get("delay_reads", 2),
                )
                self.stale_reads_remaining[composite_key] = delay_reads
                not_found_reads = pattern_match.get(
                    "not_found_reads",
                    self.eventual_consistency_config.get("not_found_reads", 0),
                )
                if not_found_reads > 0:
                    self.not_found_reads_remaining[composite_key] = not_found_reads
                _logger.debug(
                    f"Configured {delay_reads} stale reads and {not_found_reads} not-found reads for deleted item {composite_key}"
                )

                # Add to shadow index to track this deleted item
                index_key = self._get_index_key(table_name, pk)
                if index_key not in self.shadow_index:
                    self.shadow_index[index_key] = set()
                self.shadow_index[index_key].add(composite_key)
                _logger.debug(
                    f"Added {composite_key} to shadow index for eventual consistency"
                )
            else:
                _logger.debug(
                    f"No eventual consistency simulation for delete operation on {pk}:{sk}"
                )

            # Delete the item from data table and regular index
            # note: dynamdb delete is idempotent, so no error if item does not exist
            if composite_key in self.data_table:
                del self.data_table[composite_key]
                # Delete from the regular index
                index_key = self._get_index_key(table_name, pk)
                index_list = self.index_table.get(index_key, [])
                if composite_key in index_list:
                    index_list.remove(composite_key)
                    self.index_table[index_key] = index_list
                self._commit()
            else:
                _logger.debug(
                    f"Item with composite key {composite_key} does not exist, nothing to delete - but delete is idempotent in DynamoDB anyway"
                )

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
        if self.log_access:
            _logger.info(
                f"Querying table {TableName} with KeyConditionExpression: {KeyConditionExpression}, "
                f"ExpressionAttributeValues: {ExpressionAttributeValues}, "
                f"ProjectionExpression: {ProjectionExpression}, limit: {limit}"
            )

        # Validate ProjectionExpression attribute names against DynamoDB reserved keywords
        if ProjectionExpression:
            projected_keys = [
                k.strip() for k in ProjectionExpression.split(",") if k.strip()
            ]
            validate_attribute_names(
                projected_keys, f"ProjectionExpression in query for table '{TableName}'"
            )

        # Helper function to evaluate conditions
        def evaluate_condition(item, key, operator, value):
            item_value = item.get(key)
            if operator == "begins_with":
                return (item_value or "").startswith(value)
            if item_value is None:
                return False
            try:
                if operator == "=":
                    return item_value == value
                if operator == ">=":
                    return item_value >= value
                if operator == "<=":
                    return item_value <= value
                if operator == ">":
                    return item_value > value
                if operator == "<":
                    return item_value < value
            except TypeError:
                return False
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
                    projection_fields = [
                        field.strip() for field in ProjectionExpression.split(",")
                    ]

                    # Check if the item has ALL the required projection fields
                    if all(field in item for field in projection_fields):
                        projected_item = {
                            k: v for k, v in item.items() if k in projection_fields
                        }
                        results.append(projected_item)
                    # If item is missing any projection field, skip it entirely
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
        if self.log_access:
            _logger.info(
                f"Checking if item exists in table {table_name} with pk:{pk}, sk:{sk}"
            )
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
        if self.log_access:
            _logger.info(
                f"Getting paginated items from table {table_name} with limit {limit}, "
                f"key_condition_expression: {key_condition_expression}, "
                f"expression_attribute_values: {expression_attribute_values}, "
                f"projection_expression: {projection_expression}, "
                f"index_name: {index_name}"
            )
        # Check if index enforcement is enabled and index exists
        if (
            self.enforce_index_existence
            and index_name
            and (
                table_name not in self.indexes
                or index_name not in self.indexes[table_name]
            )
        ):
            raise ValueError(
                f"Index {index_name} not found. Use add_index() to define it first."
            )

        # we ignore the index and just do a full table scan
        for item in self.query(
            table_name,
            key_condition_expression,
            expression_attribute_values,
            projection_expression,
            limit=limit,
        )["Items"]:
            # Apply index projection if index is specified
            if index_name:
                item_dict = DynamoDB.item_to_dict(item)
                # For simplicity, assume table keys are 'pk' and 'sk'
                table_key_attributes = ["pk", "sk"]
                projected_item = self._apply_index_projection(
                    item_dict, table_name, index_name, table_key_attributes
                )
                try:
                    item = DynamoDB.dict_to_item(projected_item)
                except Exception as e:
                    # If serialization fails (e.g., due to Binary objects),
                    # use the original item without projection to avoid breaking functionality
                    _logger.debug(
                        f"Failed to serialize projected item due to {e}, using original item"
                    )
                    pass  # Keep the original item

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
        if self.log_access:
            _logger.info(
                f"Getting paginated items starting at pk:{pk}, sk:{sk} from table {table_name} with limit {limit}"
            )
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

    def add_index(
        self,
        table_name,
        index_name,
        projection_type,
        hash_key,
        sort_key=None,
        non_key_attributes=None,
    ):
        """
        Add an index definition to the emulator.

        Args:
            table_name: Name of the table
            index_name: Name of the index
            projection_type: One of "KEYS_ONLY", "ALL", or "INCLUDE"
            hash_key: Hash key attribute name for the index
            sort_key: Sort key attribute name for the index (optional)
            non_key_attributes: List of non-key attributes to include (only used with "INCLUDE")
        """
        # Validate projection_type
        valid_projection_types = ["KEYS_ONLY", "ALL", "INCLUDE"]
        if projection_type not in valid_projection_types:
            raise ValueError(
                f"projection_type must be one of {valid_projection_types}, got {projection_type}"
            )

        # Validate INCLUDE projection type has non_key_attributes
        if projection_type == "INCLUDE" and not non_key_attributes:
            raise ValueError(
                "projected_attributes must be provided when projection_type is INCLUDE"
            )

        if table_name not in self.indexes:
            self.indexes[table_name] = {}

        self.indexes[table_name][index_name] = {
            "projection_type": projection_type,
            "hash_key": hash_key,
            "sort_key": sort_key,
            "non_key_attributes": non_key_attributes or [],
        }

    def _apply_index_projection(
        self, item, table_name, index_name, table_key_attributes=None
    ):
        """
        Apply index projection to an item based on the projection type.

        Args:
            item: The item to project
            table_name: Name of the table
            index_name: Name of the index
            table_key_attributes: List of table's key attributes (hash and sort keys)

        Returns:
            Projected item according to the index projection type
        """
        if not self.enforce_index_existence:
            return item

        if table_name not in self.indexes or index_name not in self.indexes[table_name]:
            return item

        index_spec = self.indexes[table_name][index_name]
        projection_type = index_spec["projection_type"]

        if projection_type == "ALL":
            return item

        # Collect attributes to include in projection
        projected_attributes = set()

        # Always include index key attributes
        projected_attributes.add(index_spec["hash_key"])
        if index_spec["sort_key"]:
            projected_attributes.add(index_spec["sort_key"])

        # Always include table key attributes
        if table_key_attributes:
            projected_attributes.update(table_key_attributes)

        if projection_type == "INCLUDE":
            # Include specified non-key attributes
            projected_attributes.update(index_spec["non_key_attributes"])
        # For KEYS_ONLY, we only include the key attributes already added

        # Create projected item
        projected_item = {}
        for attr in projected_attributes:
            if attr in item:
                projected_item[attr] = item[attr]

        return projected_item

    def make_consistent(self):
        # reset stale data so that latest data is returned
        _logger.info(f"Making all reads consistent by resetting stale read counters")
        self.stale_reads_remaining = {k: 0 for k in self.stale_reads_remaining.keys()}
        self.not_found_reads_remaining = {}
        self.stale_reads_served_count = {}
        # Clear shadow index as well since all reads should now be consistent
        self.shadow_index = {}
        self.previous_versions = {}

    def get_stale_read_count(self, table_name, pk, sk):
        """Return how many stale/not-found reads were served for a specific key."""
        composite_key = self._get_composite_key(table_name, pk, sk)
        return self.stale_reads_served_count.get(composite_key, 0)
