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

                raise ValueError(
                    f"All enums must have the same attributes. {enum_key} does not have the same attributes as others."
                )
        for attr_key, value in enum_attrs.items():
            setattr(enum_key, attr_key, value)


def merge_enums(
    *enums,
    name="MergedEnum",
    base=str,
    module="__main__",
    fields=None,
    member_filter=None,
    original_enum_member_field_name=None,
):
    """Merge multiple Enum classes into a single Enum class with base type."""
    # Accumulate all members from each enum

    if fields is None:
        fields = []
    if member_filter is None:
        member_filter = lambda x: True

    new_enum_type = Enum(name, {}, module=module)

    for enum in enums:
        for member in enum:

            if not member_filter(member):
                continue

            new_value = member.value
            new_name = member.name

            # print(f"Member: {member}, value: {new_value}, name: {new_name}")

            new_member = new_enum_type._new_member_(new_enum_type)
            new_member._value_ = new_value
            value = new_member._value_

            new_member._name_ = new_name
            new_member.__objclass__ = new_enum_type.__class__
            new_member.__init__()
            new_member._values_ = (value,)
            new_member._sort_order_ = len(new_enum_type._member_names_)

            for field in fields:
                field_value = None
                if hasattr(member, field):
                    field_value = getattr(member, field)
                setattr(new_member, field, field_value)

            if original_enum_member_field_name is not None:
                setattr(new_member, original_enum_member_field_name, member)

            new_enum_type._value2member_map_[value] = new_member
            new_enum_type._member_names_.append(new_name)
            new_enum_type._member_map_[new_name] = new_member

    return new_enum_type


# Function to create a new Enum dynamically merging multiple enums
def merge_enums_2(
    enums,
    name="MergedEnum",
    base=str,
    fields=None,
    original_enum_member_field_name=None,
):
    if fields is None:
        fields = []
    """Merge multiple Enum classes into a single Enum class."""
    merged = {}
    for enum_contributor_class in enums:
        for member in enum_contributor_class:
            merged[member.name] = member.value
            # print(f"Member: {member} type: {type(member)}")
            # member_value = member
            # if hasattr(member, "value"):
            #     member_value = member.value

            # merged[member.name] = member_value
            # for field in fields:
            #     field_value = None
            #     if hasattr(enum_contributor_class, field):
            #         field_value = getattr(enum_contributor_class, field)
            #         setattr(member, field, field_value)
            # if original_enum_member_field_name is not None:
            #     setattr(member, original_enum_member_field_name,member)

    # Using the EnumMeta metaclass to create a new Enum type

    retval = EnumType(name, merged, type=base)

    return retval
