from collections import defaultdict


class DictIndex:
    """
    A class to index a list of objects (dictionaries) by given keys.

    Given a list of objects, and a list of keys, create an index such that a list of all objects matching a value for a key are found.

    Attributes:
    objects (list): A list of dictionaries to index.
    keys (list): A list of keys to index by.

    Methods:
    get(key, desired_value): Get a list of all objects matching the value for the key.


    Example:
    objects = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Alice", "age": 35}
    ]

    index = DictIndex(objects, ["name"])
    print(index.get_matching_objects("name", "Alice"))

    Output:
    [
        {"name": "Alice", "age": 30},
        {"name": "Alice", "age": 35}
    ]

    """

    def __init__(self, objects, keys, allow_missing_keys=False):
        if not isinstance(keys, list):
            raise ValueError("Keys must be a list")

        self.keys = keys

        if not isinstance(objects, list):
            raise ValueError("Objects must be a list")

        self._index = defaultdict(lambda: defaultdict(list))

        for obj in objects:
            if not isinstance(obj, dict):
                raise ValueError("Objects must be a list of dictionaries")

            for key in keys:
                if key in obj:
                    self._index[key][obj[key]].append(obj)
                elif not allow_missing_keys:
                    raise ValueError(
                        "All keys must be present in all objects. Use allow_missing_keys=True to allow missing keys."
                    )

    def get(self, key, desired_value):
        if key not in self.keys:
            raise ValueError(
                f"Objects are not indexed by {key}, index is no {self.keys}"
            )
        return self._index[key][desired_value]
