from poemai_utils.dict_index import DictIndex
from pytest import raises


def test_dict_index():

    # objects = [
    #     {"name": "Alice", "age": 30},
    #     {"name": "Bob", "age": 25},
    #     {"name": "Alice", "age": 35}
    # ]

    # index = DictIndex(objects, ["name"])
    # print(index.get("name", "Alice"))

    # Output:
    # [
    #     {"name": "Alice", "age": 30},
    #     {"name": "Alice", "age": 35}
    # ]

    objects = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Alice", "age": 35},
    ]

    index = DictIndex(objects, ["name"])
    assert index.get("name", "Alice") == [
        {"name": "Alice", "age": 30},
        {"name": "Alice", "age": 35},
    ]

    index = DictIndex(objects, ["age"])
    assert index.get("age", 30) == [{"name": "Alice", "age": 30}]
    assert index.get("age", 25) == [{"name": "Bob", "age": 25}]

    assert index.get("age", 9999) == []

    with raises(ValueError):
        index.get("uid", 9999)


def test_not_all_objects_have_keys():

    objects = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Alice"},
    ]

    with raises(ValueError):
        index = DictIndex(objects, ["age"])

    index = DictIndex(objects, ["age"], allow_missing_keys=True)
    assert index.get("age", 30) == [{"name": "Alice", "age": 30}]
