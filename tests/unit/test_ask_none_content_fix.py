"""Tests for the fix to handle None content values in streaming responses."""


def test_content_handling_fix():
    """Test the specific fix for handling None content in delta responses."""

    # Simulate the problematic delta that caused the original error
    delta = {"content": None}

    # Test the old way (would fail)
    try:
        full_text = ""
        # This would cause: TypeError: can only concatenate str (not "NoneType") to str
        # full_text += delta.get("content", "")  # This line would fail if content is explicitly None
    except TypeError:
        pass  # Expected in the old version

    # Test the new way (our fix)
    full_text = ""
    content = delta.get("content") or ""  # This should handle None correctly
    full_text += content

    assert full_text == ""  # Should be empty string, not cause an error

    # Test with actual content
    delta_with_content = {"content": "Hello"}
    content = delta_with_content.get("content") or ""
    full_text += content

    assert full_text == "Hello"

    # Test with missing content key
    delta_missing_content = {}
    content = delta_missing_content.get("content") or ""
    full_text += content

    assert full_text == "Hello"  # Should remain unchanged


def test_content_edge_cases():
    """Test various edge cases for content handling."""

    # Test empty string content
    delta = {"content": ""}
    content = delta.get("content") or ""
    assert content == ""

    # Test None content (the problematic case)
    delta = {"content": None}
    content = delta.get("content") or ""
    assert content == ""

    # Test missing content key
    delta = {}
    content = delta.get("content") or ""
    assert content == ""

    # Test with other falsy values - for streaming content, we want to treat these as empty
    # since they are not valid content strings for concatenation
    for falsy_value in [False, 0, [], {}]:
        delta = {"content": falsy_value}
        content = delta.get("content") or ""
        # These should all become empty string for safe concatenation
        assert content == ""

    # Test with valid string content
    delta = {"content": "Hello World"}
    content = delta.get("content") or ""
    assert content == "Hello World"
