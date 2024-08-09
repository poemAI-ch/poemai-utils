from enum import Enum

from poemai_utils.enum_utils import merge_enums


def test_merge_enums():

    class Enum1(str, Enum):
        A = "a"
        B = "b"

    class Enum2(str, Enum):
        C = "c"
        D = "d"

    MergedEnum = merge_enums(
        Enum1,
        Enum2,
        name="MergedEnum",
        base=str,
        module=__name__,
        fields=[],
        original_enum_member_field_name="original_enum",
    )

    members = [member for member in MergedEnum]
    member_names = set([member.name for member in members])
    member_values = set([member.value for member in members])

    assert member_names == {"A", "B", "C", "D"}
    assert member_values == {"a", "b", "c", "d"}

    # this is not working yet
    # assert MergedEnum.A == "A"
    # assert MergedEnum.B == "B"
    # assert MergedEnum.C == "C"
    # assert MergedEnum.D == "D"

    # assert MergedEnum.A.name == "A"
    # assert MergedEnum.B.name == "B"
    # assert MergedEnum.C.name == "C"
    # assert MergedEnum.D.name == "D"

    # assert MergedEnum._members_ == {
    #     "A": MergedEnum.A,
    #     "B": MergedEnum.B,
    #     "C": MergedEnum.C,
    #     "D": MergedEnum.D,
    # }
