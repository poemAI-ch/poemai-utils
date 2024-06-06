from poemai_utils.reranker import calc_rank, rerank
from pytest import raises


def test_calc_rank():
    items = ["a", "b", "c"]
    ranking_function = lambda x: ord(x) - ord("a")
    assert calc_rank(items, ranking_function) == {0: 0, 1: 1, 2: 2}

    items = ["a", "b", "c"]
    ranking_function = lambda x: abs(ord(x) - ord("c"))
    assert calc_rank(items, ranking_function) == {0: 2, 1: 1, 2: 0}


def test_rerank():
    items = [
        {"age": 20, "relevance": 10},
        {"age": 30, "relevance": 20},
        {"age": 100, "relevance": 3},
    ]

    ranking_functions_relevance = [
        (0, lambda x: x["age"]),
        (1, lambda x: x["relevance"]),
    ]

    assert rerank(items, ranking_functions_relevance) == [
        {"age": 100, "relevance": 3},
        {"age": 20, "relevance": 10},
        {"age": 30, "relevance": 20},
    ]

    ranking_functions_age = [(1, lambda x: x["age"]), (0, lambda x: x["relevance"])]

    assert rerank(items, ranking_functions_age) == [
        {"age": 20, "relevance": 10},
        {"age": 30, "relevance": 20},
        {"age": 100, "relevance": 3},
    ]

    ranking_functions_relevance_age = [
        (0.5, lambda x: x["age"]),
        (0.5, lambda x: x["relevance"]),
    ]

    assert rerank(items, ranking_functions_relevance_age) == [
        {"age": 20, "relevance": 10},
        {"age": 100, "relevance": 3},
        {"age": 30, "relevance": 20},
    ]

    with raises(ValueError):
        rerank(
            items,
            [
                (0.5, lambda x: x["age"]),
                (0.5, lambda x: x["relevance"]),
                (0.5, lambda x: x["relevance"]),
            ],
        )

    with raises(ValueError):
        rerank(items, [(0.5, lambda x: x["age"]), (0.4, lambda x: x["relevance"])])
