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
