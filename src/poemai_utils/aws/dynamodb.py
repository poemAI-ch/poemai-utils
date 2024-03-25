import json
import logging
from decimal import Clamped, Context, Inexact, Overflow, Rounded, Underflow

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

_logger = logging.getLogger(__name__)

#######################################
# Copied & adapted from boto3.dynamodb.types

# Copyright 2015 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


DYNAMODB_CONTEXT_POEMAI = Context(
    Emin=-128,
    Emax=126,
    prec=38,
    traps=[Clamped, Overflow, Inexact, Rounded, Underflow],
)
BINARY_TYPES = (bytearray, bytes)


class VersionMismatchException(Exception):
    pass


class BinaryPoemai:
    """A class for representing Binary in dynamodb

    Especially for Python 2, use this class to explicitly specify
    binary data for item in DynamoDB. It is essentially a wrapper around
    binary. Unicode and Python 3 string types are not allowed.
    """

    def __init__(self, value):
        if not isinstance(value, BINARY_TYPES):
            types = ", ".join([str(t) for t in BINARY_TYPES])
            raise TypeError(f"Value must be of the following types: {types}")
        self.value = value

    def __eq__(self, other):
        if isinstance(other, BinaryPoemai):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"Binary({self.value!r})"

    def __str__(self):
        return self.value

    def __bytes__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)


class TypeDeserializerPoemai:
    """This class deserializes DynamoDB types to Python types."""

    def deserialize(self, value):
        """The method to deserialize the DynamoDB data types.

        :param value: A DynamoDB value to be deserialized to a pythonic value.
            Here are the various conversions:

            DynamoDB                                Python
            --------                                ------
            {'NULL': True}                          None
            {'BOOL': True/False}                    True/False
            {'N': str(value)}                       Decimal(str(value)) or int, if value is an integer
            {'S': string}                           string
            {'B': bytes}                            Binary(bytes)
            {'NS': [str(value)]}                    set([Decimal(str(value))])
            {'SS': [string]}                        set([string])
            {'BS': [bytes]}                         set([bytes])
            {'L': list}                             list
            {'M': dict}                             dict

        :returns: The pythonic value of the DynamoDB type.
        """

        if not value:
            raise TypeError(
                "Value must be a nonempty dictionary whose key "
                "is a valid dynamodb type."
            )
        dynamodb_type = list(value.keys())[0]
        try:
            deserializer = getattr(self, f"_deserialize_{dynamodb_type}".lower())
        except AttributeError:
            raise TypeError(f"Dynamodb type {dynamodb_type} is not supported")
        return deserializer(value[dynamodb_type])

    def _deserialize_null(self, value):
        return None

    def _deserialize_bool(self, value):
        return value

    def _deserialize_n(self, value):
        try:
            return int(value)
        except ValueError:
            return DYNAMODB_CONTEXT_POEMAI.create_decimal(value)

    def _deserialize_s(self, value):
        return value

    def _deserialize_b(self, value):
        return BinaryPoemai(value)

    def _deserialize_ns(self, value):
        return set(map(self._deserialize_n, value))

    def _deserialize_ss(self, value):
        return set(map(self._deserialize_s, value))

    def _deserialize_bs(self, value):
        return set(map(self._deserialize_b, value))

    def _deserialize_l(self, value):
        return [self.deserialize(v) for v in value]

    def _deserialize_m(self, value):
        return {k: self.deserialize(v) for k, v in value.items()}


# END COPY


class DynamoDB:
    ddb_type_deserializer = TypeDeserializerPoemai()
    ddb_type_serializer = TypeSerializer()

    def __init__(self, config):
        _logger.info(
            f"Initializing DynamoDB with config: REGION_NAME={config.REGION_NAME}"
        )
        self.region_name = config.REGION_NAME
        self.dynamodb_resource = boto3.resource(
            "dynamodb", region_name=self.region_name
        )
        self.dynamodb_client = boto3.client("dynamodb", region_name=self.region_name)

    def store_item(self, table_name, item):
        dynamodb_item = self.ddb_type_serializer.serialize(item)
        _logger.debug("Storing item %s in table %s", item, table_name)
        _logger.debug("Serialized to %s", json.dumps(dynamodb_item, indent=2))
        dynamodb_item = dynamodb_item["M"]
        response = self.put_item(
            TableName=table_name,
            Item=dynamodb_item,
        )
        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(f"Error storing item {item}, response: {response}")
        else:
            _logger.debug(f"Stored item {item}, response: {response}")

    def update_versioned_item_by_pk_sk(
        self,
        table_name,
        pk,
        sk,
        attribute_updates,
        expected_version,
        version_attribute_name="version",
    ):
        # Build the update expression
        set_expressions = []
        expression_attribute_values = {":expectedVersion": {"N": str(expected_version)}}

        # Increment the version
        set_expressions.append(f"#{version_attribute_name} = :newVersion")
        expression_attribute_values[":newVersion"] = {"N": str(expected_version + 1)}

        # Add other attributes to the update expression
        for attr, value in attribute_updates.items():
            placeholder = f":{attr}"
            set_expressions.append(f"#{attr} = {placeholder}")
            expression_attribute_values[placeholder] = (
                self.ddb_type_serializer.serialize(value)
            )

        update_expression = "SET " + ", ".join(set_expressions)
        expression_attribute_names = {
            f"#{attr}": attr for attr in attribute_updates.keys()
        }
        expression_attribute_names[f"#{version_attribute_name}"] = (
            version_attribute_name
        )

        try:
            # Perform a conditional update
            response = self.dynamodb_client.update_item(
                TableName=table_name,
                Key={"pk": {"S": pk}, "sk": {"S": sk}},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                ConditionExpression=f"#{version_attribute_name} = :expectedVersion",
                ReturnValues="UPDATED_NEW",
            )
            _logger.debug(
                f"Updated item {pk}:{sk} in table {table_name}, response: {response}"
            )
            return response
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                _logger.error(
                    f"Update failed: Optimistic lock failed, version mismatch for item {pk}:{sk}, expected {expected_version}"
                )
                raise VersionMismatchException(
                    f"Version mismatch updating {pk}:{sk}, expecting {expected_version}"
                ) from e
            else:
                _logger.error(
                    f"Update failed for item {pk}:{sk}, response: {e.response}"
                )
                raise

    def put_item(self, TableName, Item):
        """A proxy for boto3.dynamodb.table.put_item"""

        response = self.dynamodb_client.put_item(
            TableName=TableName,
            Item=Item,
        )

        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(f"Error storing item {Item}, response: {response}")
        else:
            _logger.debug(f"Stored item {Item}, response: {response}")

        return response

    def get_item_by_pk_sk(self, table_name, pk, sk):
        response = self.get_item(
            TableName=table_name,
            Key={
                "pk": {"S": pk},
                "sk": {"S": sk},
            },
        )
        if "Item" in response:
            return self.item_to_dict(response["Item"])
        else:
            return None

    def batch_get_items_by_pk_sk(self, table_name, pk_sk_list):
        db_keys = [{"pk": i["pk"], "sk": i["sk"]} for i in pk_sk_list]
        if len(db_keys) == 0:
            return []

        # split into chunks of 100
        db_keys_chunks = [db_keys[i : i + 100] for i in range(0, len(db_keys), 100)]
        for db_keys_chunk in db_keys_chunks:
            response = self.batch_get_item(
                RequestItems={table_name: {"Keys": db_keys_chunk}}
            )
            for item in response["Responses"][table_name]:
                yield self.item_to_dict(item)

    def get_item_by_pk(self, table_name, pk):
        response = self.get_item(
            TableName=table_name,
            Key={
                "pk": {"S": pk},
            },
        )
        if "Item" in response:
            return self.item_to_dict(response["Item"])
        else:
            return None

    def scan_for_items(
        self,
        table_name,
        filter_expression,
        expression_attribute_values,
        projection_expression=None,
    ):
        paginator = self.dynamodb_client.get_paginator("scan")
        args = {"TableName": table_name}
        if filter_expression is not None:
            args["FilterExpression"] = filter_expression
        if expression_attribute_values is not None:
            args["ExpressionAttributeValues"] = expression_attribute_values
        if projection_expression is not None:
            args["ProjectionExpression"] = projection_expression
        page_iterator = paginator.paginate(**args)

        for page in page_iterator:
            for item in page["Items"]:
                yield self.item_to_dict(item)

    def scan_for_items_by_pk_sk(self, table_name, pk_contains, sk_contains):
        filter_expression = ""
        if pk_contains is not None:
            filter_expression += "contains(pk, :pk)"
        if sk_contains is not None:
            if filter_expression != "":
                filter_expression += " and "
            filter_expression += "contains(sk, :sk)"

        expression_attribute_values = {}
        if pk_contains is not None:
            expression_attribute_values[":pk"] = {"S": pk_contains}
        if sk_contains is not None:
            expression_attribute_values[":sk"] = {"S": sk_contains}

        for item in self.scan_for_items(
            table_name, filter_expression, expression_attribute_values
        ):
            yield item

    def delete_item_by_pk_sk(self, table_name, pk, sk):
        response = self.delete_item(
            TableName=table_name,
            Key={
                "pk": {"S": pk},
                "sk": {"S": sk},
            },
        )
        return response

    def delete_item(self, TableName, Key):
        """A proxy for boto3.dynamodb.table.delete_item"""

        response = self.dynamodb_client.delete_item(
            TableName=TableName,
            Key=Key,
        )

        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(f"Error deleting item {Key}, response: {response}")
        else:
            _logger.debug(f"Deleted item {Key}, response: {response}")

        return response

    def get_paginated_items(
        self,
        table_name,
        key_condition_expression,
        expression_attribute_values,
        projection_expression=None,
        limit=100,
    ):
        """A proxy for boto3.dynamodb.table.query"""

        paginator = self.dynamodb_client.get_paginator("query")
        if projection_expression is not None:
            page_iterator = paginator.paginate(
                TableName=table_name,
                KeyConditionExpression=key_condition_expression,
                ExpressionAttributeValues=expression_attribute_values,
                ProjectionExpression=projection_expression,
                Limit=limit,
            )
        else:
            page_iterator = paginator.paginate(
                TableName=table_name,
                KeyConditionExpression=key_condition_expression,
                ExpressionAttributeValues=expression_attribute_values,
                Limit=limit,
            )

        for page in page_iterator:
            for item in page["Items"]:
                yield item

    def get_paginated_items_by_pk(self, table_name, pk, limit=100):
        for item in self.get_paginated_items(
            table_name=table_name,
            key_condition_expression="pk = :pk",
            expression_attribute_values={":pk": {"S": pk}},
            limit=limit,
        ):
            yield self.item_to_dict(item)

    def get_item(self, TableName, Key):
        """A proxy for boto3.dynamodb.table.get_item"""

        _logger.debug(f"Getting item Key={Key} from table TableName={TableName}")
        response = self.dynamodb_client.get_item(
            TableName=TableName,
            Key=Key,
        )

        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error("Error getting item %s, response: %s", Key, response)
        else:
            _logger.debug("Got item %s, response: %s", Key, response)

        return response

    def query(
        self,
        TableName,
        KeyConditionExpression,
        ExpressionAttributeValues,
        ProjectionExpression=None,
    ):
        """A proxy for boto3.dynamodb.table.query"""

        args = {"TableName": TableName}
        if KeyConditionExpression is not None:
            args["KeyConditionExpression"] = KeyConditionExpression
        if ExpressionAttributeValues is not None:
            args["ExpressionAttributeValues"] = ExpressionAttributeValues
        if ProjectionExpression is not None:
            args["ProjectionExpression"] = ProjectionExpression

        response = self.dynamodb_client.query(**args)

        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(
                f"Error running query with KeyConditionExpression {KeyConditionExpression}, ExpressionAttributeValues {ExpressionAttributeValues}, response: {response}"
            )

        return response

    def batch_get_item(self, RequestItems):
        """A proxy for boto3.dynamodb.table.batch_get_item"""

        response = self.dynamodb_client.batch_get_item(RequestItems=RequestItems)

        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(
                f"Error running batch_get_item with RequestItems {RequestItems}, response: {response}"
            )

        return response

    def batch_write_item(self, RequestItems):
        """A proxy for boto3.dynamodb.table.batch_write_item"""

        response = self.dynamodb_client.batch_write_item(RequestItems=RequestItems)

        # check response for errors
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(
                f"Error running batch_write_item with RequestItems {RequestItems}, response: {response}"
            )

        return response

    def batch_write(self, table_name, object_list):
        put_requests = []
        for obj in object_list:
            put_requests.append(
                {
                    "PutRequest": {
                        "Item": self.dict_to_item(obj),
                    }
                }
            )
        request_items = {table_name: put_requests}
        response = self.batch_write_item(RequestItems=request_items)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _logger.error(
                f"Error running batch_write_item with RequestItems {request_items}, response: {response}"
            )

        return response

    def item_exists(self, table_name, pk, sk):
        try:
            response = self.dynamodb_client.get_item(
                TableName=table_name,
                Key={
                    "pk": {"S": pk},
                    "sk": {"S": sk},
                },
            )
            if "Item" in response:
                return True
            return False
        except self.dynamodb_client.exceptions.ResourceNotFoundException:
            return False

    @classmethod
    def item_to_dict(cls, item):
        if item is None:
            return {}
        return cls.ddb_type_deserializer.deserialize({"M": item})

    @classmethod
    def dict_to_item(cls, d):
        return cls.ddb_type_serializer.serialize(d)["M"]

    @classmethod
    def pk_sk_from_fields(cls, pk_items, sk_items):
        pk = "#".join([f"{k}#{v}" for k, v in pk_items])
        sk = "#".join([f"{k}#{v}" for k, v in sk_items])
        return pk, sk

    @classmethod
    def pk_sk_fields(cls, pk, sk):
        pk_split = pk.split("#")
        # build pairs from pk_pksplit
        pk_pairs = zip(pk_split[::2], pk_split[1::2])
        # build dict from pairs
        pk_dict = {k.lower(): v for k, v in pk_pairs}

        sk_split = sk.split("#")
        # build pairs from sk_pksplit
        sk_pairs = zip(sk_split[::2], sk_split[1::2])
        # build dict from pairs
        sk_dict = {k.lower(): v for k, v in sk_pairs}
        all_keys = {**pk_dict, **sk_dict}
        # _logger.info(f"pk_sk_fields: {all_keys}")
        return all_keys
