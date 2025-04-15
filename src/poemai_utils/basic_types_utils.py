from decimal import Decimal


def linebreak(text, max_length=80):
    """Add linebreaks to text to make it more readable"""

    out_lines = []
    for line in text.split("\n"):
        if len(line) <= max_length:
            out_lines.append(line)
        else:
            # split line into words
            words = line.split(" ")
            out_line = ""
            for word in words:
                if len(out_line) + len(word) > max_length:
                    out_lines.append(out_line)
                    out_line = ""
                out_line += word + " "
            out_lines.append(out_line)
    return "\n".join(out_lines)


def short_display(text, max_lenght=100):
    if len(text) > max_lenght:
        return text[:max_lenght] + "..."
    return text


def remove_none_from_dict(
    dictionary, remove_empty_lists=False, remove_empty_dicts=False
):
    retval = {}

    for key, value in list(dictionary.items()):
        if value is None:
            continue
        elif isinstance(value, dict):
            new_value = remove_none_from_dict(value)
            if not new_value and remove_empty_dicts:
                continue
            else:
                retval[key] = new_value
        elif isinstance(value, list):
            if len(value) == 0 and remove_empty_lists:
                continue
            else:
                retval[key] = [
                    remove_none_from_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
        else:
            retval[key] = value

    return retval


def compare_as_strings(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return str(a) == str(b)


def any_to_bool(value):
    """
    Converts a value to a boolean. Handles common string representations
    of boolean values as well as numeric values.

    Parameters:
    value (str/int/float/bool/None): The value to convert.

    Returns:
    bool: The boolean representation of the value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    return False  # Default to False for any other cases (None, empty string, etc.)


def replace_floats_with_decimal(obj):
    """
    Recursively replaces float values in nested structures (lists, dictionaries)
    with Decimal representations.
    """
    if isinstance(obj, dict):
        return {key: replace_floats_with_decimal(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_floats_with_decimal(element) for element in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj


def replace_decimal_with_string(obj):
    """
    Recursively replaces Decimal values in nested structures (lists, dictionaries)
    with string representations, so that json serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: replace_decimal_with_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_decimal_with_string(element) for element in obj]
    elif isinstance(obj, Decimal):
        return str(obj)
    else:
        return obj
