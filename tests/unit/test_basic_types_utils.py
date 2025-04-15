from decimal import Decimal

from poemai_utils.basic_types_utils import replace_floats_with_decimal


def test_replace_floats_with_decimal():

    nested_dict = {
        "key1": 1.5,
        "subdict": {
            "key2": 2.5,
            "subsubdict": {
                "key3": 3.5,
                "list": [4.5, 5.5],
            },
        },
        "list": [6.5, 7.5],
        "key4": "string",
        "key5": None,
        "key6": Decimal(99.99),
    }
    expected_result = {
        "key1": Decimal("1.5"),
        "subdict": {
            "key2": Decimal("2.5"),
            "subsubdict": {
                "key3": Decimal("3.5"),
                "list": [Decimal("4.5"), Decimal("5.5")],
            },
        },
        "list": [Decimal("6.5"), Decimal("7.5")],
        "key4": "string",
        "key5": None,
        "key6": Decimal(99.99),
    }
    result = replace_floats_with_decimal(nested_dict)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
