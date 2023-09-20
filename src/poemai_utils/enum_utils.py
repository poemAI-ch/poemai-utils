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
    for enum_key, enum_attrs in attr_dict.items():
        for attr_key, value in enum_attrs.items():
            setattr(enum_key, attr_key, value)
