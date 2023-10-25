from poemai_utils.openai.ask import Ask


def test_count_tokens():
    ask = Ask()
    assert ask.count_tokens("hello world") == 2
