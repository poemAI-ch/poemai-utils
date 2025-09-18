from enum import Enum

import pytest
from poemai_utils.ai_model import AIApiType, AIModel
from poemai_utils.enum_utils import add_enum_attrs


@pytest.fixture(autouse=True)
def clear_ai_model():
    # Automatically clears AIModel before each test
    AIModel._clear()


class OPENAI_MODEL(Enum):
    GPT_4_o = "gpt-4o"
    GPT_4_o_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4_o_2024_08_06 = "gpt-4o-2024-08-06"


def test_find_model_by_various_keys():
    # Add models to the directory
    AIModel.add_enum_members(OPENAI_MODEL, "openai")

    # Test retrieval by exact name
    model = AIModel.find_model("GPT_4_o_2024_05_13")
    assert model.name == "GPT_4_o_2024_05_13"

    # Test retrieval by model_key attribute if set
    setattr(model, "model_key", "gpt-4o-2024-05-13")
    found_model = AIModel.find_model("gpt-4o-2024-05-13")
    assert found_model == model

    # Test retrieval by case-insensitive name
    model = AIModel.find_model("gpt_4_o_2024_05_13")
    assert model.name == "GPT_4_o_2024_05_13"

    # Test retrieval by value if the enum uses a different identifier
    setattr(model, "value", "gpt-4o-may")
    found_model = AIModel.find_model("gpt-4o-may")
    assert found_model == model

    # Test for unknown model_key
    with pytest.raises(ValueError) as excinfo:
        AIModel.find_model("unknown_model_key")
    assert "Unknown model_key: unknown_model_key" in str(excinfo.value)


def test_model_retrieval_with_dot_notation():
    # Add models to the directory
    AIModel.add_enum_members(OPENAI_MODEL, "openai")

    # Simulating model_key access with a namespace
    model = AIModel.find_model("openai.GPT_4_o_2024_05_13")
    assert model.name == "GPT_4_o_2024_05_13"

    # Ensuring the split happens correctly and still retrieves the model
    model = AIModel.find_model("something.else.GPT_4_o_2024_05_13")
    assert model.name == "GPT_4_o_2024_05_13"


def test_adding_openai_models():
    AIModel.add_enum_members(OPENAI_MODEL, "openai")
    assert len(list(AIModel.get_realm_members("openai"))) == 3
    # Check for idempotence by adding the same enum again
    AIModel.add_enum_members(OPENAI_MODEL, "openai")
    assert len(list(AIModel.get_realm_members("openai"))) == 3


def test_adding_dict_based_realm():
    models_list = [
        {"name": "CUSTOM_MODEL_1", "feature": "feature1"},
        {"name": "CUSTOM_MODEL_2", "feature": "feature2"},
    ]
    AIModel.register_realm(models_list, "custom_realm")
    assert len(list(AIModel.get_realm_members("custom_realm"))) == 2


def test_adding_same_realm_twice_unmodified():
    models_list = [
        {"name": "STABLE_MODEL_1", "feature": "stable1"},
        {"name": "STABLE_MODEL_2", "feature": "stable2"},
    ]
    AIModel.register_realm(models_list, "stable_realm")
    # Try to register the same realm with the same data again
    AIModel.register_realm(models_list, "stable_realm")
    assert len(list(AIModel.get_realm_members("stable_realm"))) == 2


def test_adding_same_realm_twice_modified():
    models_list_initial = [{"name": "DYNAMIC_MODEL_1", "feature": "dynamic1"}]
    models_list_modified = [
        {"name": "DYNAMIC_MODEL_1", "feature": "dynamic1"},
        {"name": "DYNAMIC_MODEL_2", "feature": "dynamic2"},
    ]
    AIModel.register_realm(models_list_initial, "dynamic_realm")
    with pytest.raises(ValueError):
        AIModel.register_realm(models_list_modified, "dynamic_realm")


def test_accessing_model_by_subscript():
    AIModel.add_enum_members(OPENAI_MODEL, "openai")
    model = AIModel["GPT_4_o_2024_05_13"]
    assert model.name == "GPT_4_o_2024_05_13"
    assert model.realm_id == "openai"


def test_iterating_through_models():
    AIModel.add_enum_members(OPENAI_MODEL, "openai")
    names = [model.name for model in AIModel]
    expected_names = ["GPT_4_o", "GPT_4_o_2024_05_13", "GPT_4_o_2024_08_06"]
    assert set(names) == set(expected_names)


def test_accessing_member_as_attribute():
    AIModel.add_enum_members(OPENAI_MODEL, "openai")
    model = AIModel.GPT_4_o_2024_05_13
    assert model.name == "GPT_4_o_2024_05_13"
    assert model.realm_id == "openai"


def test_model_hashability():
    AIModel.add_enum_members(OPENAI_MODEL, "openai")
    model_set = set()
    model_dict = {}

    model = AIModel["GPT_4_o_2024_05_13"]
    model_set.add(model)
    model_dict[model] = "Test Entry"

    assert model in model_set
    assert model_dict[model] == "Test Entry"
    assert len(model_set) == 1
    assert len(model_dict) == 1

    # Adding the same model should not increase the size of the set or dict
    model_set.add(model)
    model_dict[model] = "Updated Entry"
    assert len(model_set) == 1
    assert len(model_dict) == 1
    assert model_dict[model] == "Updated Entry"


def test_embedding_check():
    add_enum_attrs(
        {
            OPENAI_MODEL.GPT_4_o: {
                "model_key": "gpt-4o",
                "api_types": [AIApiType.EMBEDDINGS],
                "expensive": False,
                "supports_vision": True,
                "embeddings_dimensions": None,
            },
            OPENAI_MODEL.GPT_4_o_2024_05_13: {
                "model_key": "gpt-4o-2024-05-13",
                "api_types": [AIApiType.CHAT_COMPLETIONS],
                "expensive": False,
                "supports_vision": True,
                "embeddings_dimensions": None,
            },
            OPENAI_MODEL.GPT_4_o_2024_08_06: {
                "model_key": "gpt-4o-2024-08-06",
                "api_types": [AIApiType.CHAT_COMPLETIONS],
                "expensive": False,
                "supports_vision": True,
                "embeddings_dimensions": None,
            },
        }
    )

    AIModel.add_enum_members(OPENAI_MODEL, "openai")

    assert AIModel.GPT_4_o.is_embedding_model()
    assert not AIModel.GPT_4_o_2024_05_13.is_embedding_model()
    assert not AIModel.GPT_4_o_2024_08_06.is_embedding_model()


def test_custom_embedding_model():

    custom_model = {"name": "CUSTOM_EMBEDDING_MODEL", "api_types": ["EMBEDDINGS"]}

    AIModel.register_realm([custom_model], "custom_realm")

    assert AIModel.CUSTOM_EMBEDDING_MODEL.is_embedding_model()

    custom_model_2 = {
        "name": "CUSTOM_EMBEDDING_MODEL_2",
        "api_types": ["AIApiType.embeddings"],
    }

    AIModel.register_realm([custom_model_2], "custom_realm_2")

    assert AIModel.CUSTOM_EMBEDDING_MODEL_2.is_embedding_model()
