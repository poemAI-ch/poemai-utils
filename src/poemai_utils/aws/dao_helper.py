import hashlib
import logging
from typing import Any, Dict, List, Optional

from poemai_utils.aws.dynamodb import DynamoDB

_logger = logging.getLogger(__name__)


class DaoHelper:

    @staticmethod
    def build_key(
        key_element_enum,
        field_to_key_formatters,
        key_components: List[Any],
        values: Dict[str, Any],
    ) -> str:
        def transform_key(key):
            if key == key_element_enum.OBJECT_TYPE:
                return values["object_type"]
            return key.name

        def fetch_value(key):
            if key == key_element_enum.OBJECT_TYPE:
                return ""
            raw_value = values.get(key.name.lower(), "")
            if raw_value in (None, ""):
                return ""
            formatter = field_to_key_formatters.get(key)
            return formatter["formatter"](raw_value) if formatter else raw_value

        parts = [(k, fetch_value(k)) for k in key_components]
        return "#".join([f"{transform_key(k)}#{v}" for k, v in parts])

    @classmethod
    def build_pk_sk_from_components(
        cls,
        key_element_enum,
        field_to_key_formatters,
        pk_components,
        sk_components,
        values,
    ):
        pk = cls.build_key(
            key_element_enum, field_to_key_formatters, pk_components, values
        )
        sk = (
            cls.build_key(
                key_element_enum, field_to_key_formatters, sk_components, values
            )
            if sk_components
            else None
        )
        return pk, sk

    @classmethod
    def build_pk_sk(
        cls,
        key_element_enum,
        field_to_key_formatters,
        object_type,
        values: Dict[str, Any],
    ):
        values["object_type"] = object_type.name
        return cls.build_pk_sk_from_components(
            key_element_enum,
            field_to_key_formatters,
            object_type.pk_components,
            object_type.sk_components,
            values,
        )

    @classmethod
    def build_pk_sk_dict(
        cls,
        key_element_enum,
        field_to_key_formatters,
        object_type,
        values: Dict[str, Any],
    ) -> Dict[str, str]:
        pk, sk = cls.build_pk_sk(
            key_element_enum, field_to_key_formatters, object_type, values
        )
        return {"pk": pk, "sk": sk}

    @classmethod
    def build_pk_sk_item(
        cls, key_element_enum, field_to_key_formatters, object_type, values
    ):
        return DynamoDB.dict_to_item(
            cls.build_pk_sk_dict(
                key_element_enum, field_to_key_formatters, object_type, values
            )
        )

    @classmethod
    def check_required_fields(cls, object_type, values):
        for required in object_type.required_fields:
            if required.name.lower() not in values:
                raise ValueError(f"Missing required field {required.name}")

    @classmethod
    def pk_sk_fields_of_object(
        cls, key_element_enum, key_to_field_formatters, object_type, pk: str, sk: str
    ):

        key_fields = DynamoDB.pk_sk_fields(pk, sk)
        key_components = [
            kc
            for kc in object_type.pk_components + object_type.sk_components
            if kc != key_element_enum.OBJECT_TYPE
        ]

        result = {}
        for key in key_components:
            name = key.name.lower()
            raw = key_fields.get(name)
            if raw is not None:
                transform = key_to_field_formatters.get(key)
                result[name] = transform["formatter"](raw) if transform else raw
        return result

    @classmethod
    def get_object(
        cls,
        key_element_enum,
        field_to_key_formatters,
        db,
        table_name,
        object_type,
        values,
    ):
        cls.check_required_fields(object_type, values)
        pk, sk = cls.build_pk_sk(
            key_element_enum, field_to_key_formatters, object_type, values
        )
        return db.get_item_by_pk_sk(table_name, pk, sk)

    @classmethod
    def get_object_with_pk_sk_fields(
        cls,
        key_element_enum,
        field_to_key_formatters,
        key_to_field_formatters,
        db,
        table_name,
        object_type,
        values,
    ):
        raw_object = cls.get_object(
            key_element_enum,
            field_to_key_formatters,
            db,
            table_name,
            object_type,
            values,
        )

        if raw_object is None:
            return None

        fields = cls.pk_sk_fields_of_object(
            key_element_enum,
            key_to_field_formatters,
            object_type,
            raw_object["pk"],
            raw_object.get("sk"),
        )
        return {**raw_object, **fields}

    @staticmethod
    def calc_content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def store_object(
        cls,
        key_element_enum,
        field_to_key_formatters,
        db,
        table_name,
        object_type,
        values,
        versioned=False,
        only_if_new=False,
    ):
        cls.check_required_fields(object_type, values)
        pk, sk = cls.build_pk_sk(
            key_element_enum, field_to_key_formatters, object_type, values
        )

        to_drop_field_keys = [k.name.lower() for k in object_type.to_drop_fields]
        values_to_store = {
            k: v for k, v in values.items() if k not in to_drop_field_keys
        }
        _logger.debug(f"Storing object {object_type} with values {values_to_store}")

        if versioned:
            if "version" in values_to_store:
                del values_to_store["version"]
            values_to_store.pop("version", None)
            values_to_store.pop("pk", None)
            values_to_store.pop("sk", None)

            _logger.debug(
                f"Storing versioned object {object_type} with values {values_to_store}"
            )
            db.update_versioned_item_by_pk_sk(
                table_name, pk, sk, values_to_store, values["version"]
            )
        else:
            item = {
                **values_to_store,
                "pk": pk,
            }
            if sk is not None:
                item["sk"] = sk

            if only_if_new:
                db.store_new_item(
                    table_name, item, "sk"
                )  # works because dynamodb checks for uniqueness of pk and sk
            else:
                db.store_item(table_name, item)

        return {"pk": pk, "sk": sk}

    @classmethod
    def delete_object(
        cls,
        key_element_enum,
        field_to_key_formatters,
        db,
        table_name,
        object_type,
        values,
    ):
        pk, sk = cls.build_pk_sk(
            key_element_enum, field_to_key_formatters, object_type, values
        )
        db.delete_item_by_pk_sk(table_name, pk, sk)

    @classmethod
    def object_exists(
        cls,
        key_element_enum,
        field_to_key_formatters,
        db,
        table_name,
        object_type,
        values,
    ):
        pk, sk = cls.build_pk_sk(
            key_element_enum, field_to_key_formatters, object_type, values
        )
        return db.item_exists(table_name, pk, sk)
