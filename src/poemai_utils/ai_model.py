from enum import Enum

from poemai_utils.enum_utils import add_enum_repr


class AIApiType(Enum):
    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"
    EMBEDDINGS = "embeddings"
    MODERATIONS = "moderations"


add_enum_repr(AIApiType)


class ModelMeta(type):
    def __iter__(cls):
        return iter(cls._member_dict.values())


class AIModel(metaclass=ModelMeta):

    _member_dict = {}
    _initialized_realm_ids = set()

    def __init__(self, name, realm_id, **kwargs):
        self.realm_id = realm_id
        self.name = name

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._member_dict[self.name] = self
        setattr(AIModel, self.name, self)

    def __str__(self):
        return f"AIModel.{self.name}"

    def __repr__(self):
        return f"AIModel.{self.name}"

    def __class_getitem__(cls, item):
        return cls._member_dict[item]

    def __eq__(self, other):
        if isinstance(other, AIModel):
            return (self.name, self.realm_id) == (other.name, other.realm_id)
        return False

    def __hash__(self):
        return hash((self.name, self.realm_id))

    def is_embedding_model(self):
        if hasattr(self, "api_types"):
            return AIApiType.EMBEDDINGS in self.api_types
        return False

    @classmethod
    def __next__(cls):
        return next(iter(cls._member_dict.values()))

    @classmethod
    def get_realm_members(cls, realm_id):
        return [model for model in cls if model.realm_id == realm_id]

    @classmethod
    def get_realms(cls):
        return set([model.realm_id for model in cls])

    @classmethod
    def add_enum_members(cls, enum_class, realm_id):

        if realm_id in cls._initialized_realm_ids:
            realm_member_names = set(
                [model.name for model in cls.get_realm_members(realm_id)]
            )
            new_member_names = set([member.name for member in enum_class])

            if realm_member_names == new_member_names:
                # nothing changed - make this idempotent
                return

            raise ValueError(
                f"Realm id {realm_id} is already initialized, and the new members do not match the existing members"
            )
        for enum_member in enum_class:
            additional_attrs = {
                k: v
                for k, v in vars(enum_member).items()
                if not k.startswith("_") and not k == "name"
            }
            if "api_types" in additional_attrs:
                additional_attrs["api_types"] = cls._convert_to_api_types(
                    additional_attrs["api_types"]
                )

            _ = AIModel(enum_member.name, realm_id, **additional_attrs)

        cls._initialized_realm_ids.add(realm_id)

    @classmethod
    def register_realm(cls, models_list, realm_id):
        if realm_id in cls._initialized_realm_ids:

            realm_member_names = set(
                [model.name for model in cls.get_realm_members(realm_id)]
            )
            new_member_names = set([member["name"] for member in models_list])

            if realm_member_names == new_member_names:
                # nothing changed - make this idempotent
                return

            raise ValueError(
                f"Realm id {realm_id} is already initialized, and the new members do not match the existing members"
            )

        for model in models_list:
            name = model["name"]
            rest_attrs = {k: v for k, v in model.items() if k != "name"}
            if "api_types" in rest_attrs:
                rest_attrs["api_types"] = cls._convert_to_api_types(
                    rest_attrs["api_types"]
                )

            if name in cls._member_dict:
                raise KeyError(f"Model {name} is already registered in the directory.")
            model = AIModel(name, realm_id, **rest_attrs)
            cls._member_dict[name] = model
            setattr(cls, name, model)

        cls._initialized_realm_ids.add(realm_id)

    @classmethod
    def find_model(cls, model_key: str) -> "AIModel":
        if "." in model_key:
            model_key = model_key.split(".")[-1]
        for model in cls:
            if model.name == model_key:
                return model
            if hasattr(model, "model_key") and model.model_key == model_key:
                return model
            if hasattr(model, "value") and model.value == model_key:
                return model
            if model.name.lower() == model_key.lower():
                return model
        raise ValueError(f"Unknown model_key: {model_key}")

    @classmethod
    def _clear(cls):
        cls._member_dict = {}
        cls._initialized_realm_ids = set()

    @classmethod
    def _convert_to_api_types(cls, api_types):
        retval = []
        for api_type in api_types:
            if isinstance(api_type, AIApiType):
                retval.append(api_type)
            else:
                if "." in api_type:
                    api_type = api_type.split(".")[-1]
                api_type = api_type.upper()
                retval.append(AIApiType[api_type])
        return retval
