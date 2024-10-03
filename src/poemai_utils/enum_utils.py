from enum import Enum, EnumType


def _enum_str(self):
    return f"{self.__class__.__name__}.{self.name}"


def _enum_str_attr(self):
    # format an enum such that it is valid python code
    additional_attrs = []
    for k, v in vars(self).items():
        if not k.startswith("_"):
            if isinstance(v, str):
                additional_attrs.append((k, f"'{v}'"))
            else:
                additional_attrs.append((k, v))
    if len(additional_attrs) > 0:
        return f"{self.__class__.__name__}.{self.name}({','.join([f'{k}={v}' for k,v in additional_attrs])})"
    return f"{self.__class__.__name__}.{self.name}"


def _enum_repr(self):
    return _enum_str(self)


def add_enum_repr(enum_class):
    enum_class.__str__ = _enum_str
    enum_class.__repr__ = _enum_repr
    return enum_class


def add_enum_repr_attr(enum_class):
    enum_class.__str__ = _enum_str_attr
    enum_class.__repr__ = _enum_str
    return enum_class


def add_enum_attrs(attr_dict):
    attr_set = None
    enum_classes = set([k.__class__ for k in attr_dict.keys()])
    if len(enum_classes) > 1:
        raise ValueError("All enums must be of the same class")

    enum_class = enum_classes.pop()
    members = set([e for e in enum_class])

    enum_names_keys = set(attr_dict.keys())

    if members != enum_names_keys:
        missing_names = [e.name for e in members - enum_names_keys]
        raise ValueError(
            f"All enums must be defined in the enum class, but {missing_names} are missing."
        )

    for enum_key, enum_attrs in attr_dict.items():
        if attr_set is None:
            attr_set = set(enum_attrs.keys())
        else:
            if set(enum_attrs.keys()) != attr_set:

                missing = attr_set - set(enum_attrs.keys())
                too_much = set(enum_attrs.keys()) - attr_set

                raise ValueError(
                    f"All enums must have the same attributes. {enum_key} does not have the same attributes as others, missing: {missing}, new: {too_much}"
                )
        for attr_key, value in enum_attrs.items():
            setattr(enum_key, attr_key, value)
