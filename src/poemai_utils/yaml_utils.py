from io import StringIO


def dict_to_yaml_string(data):
    """
    Converts a dictionary to a YAML string, ensuring multiline strings use the '|' block style.

    Args:
        data: The input data dictionary.

    Returns:
        A string containing the YAML representation of the input data.
    """
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.scalarstring import LiteralScalarString

    except ImportError:
        raise ImportError(
            "Please install the 'ruamel.yaml' package to use this function."
        )

    def process_multiline_strings(data):
        """
        Recursively processes the input data, wrapping multiline strings in LiteralScalarString.

        Args:
            data: The input data (dict, list, or string).

        Returns:
            The processed data with multiline strings wrapped.
        """

        if isinstance(data, dict):
            return {k: process_multiline_strings(v) for k, v in data.items()}
        elif isinstance(data, str):
            if "\n" in data:
                return LiteralScalarString(data)
            else:
                return data
        elif isinstance(data, list):
            return [process_multiline_strings(v) for v in data]
        else:
            return data

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.allow_unicode = True

    processed_data = process_multiline_strings(data)

    stream = StringIO()
    yaml.dump(processed_data, stream)
    return stream.getvalue()
