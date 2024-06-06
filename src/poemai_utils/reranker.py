import numpy as np


def calc_rank(items, ranking_function):
    """Returns a dict where the key is the index of the item in the list, and the value is the rank of the item.
    The rank is calculated by the ranking_function, which takes an item and returns a number.
    The best item has rank 0, the second best rank 1, and so on.

    Example:
    items = ["a", "b", "c"]
    ranking_function = lambda x: ord(x) - ord("a")
    calc_rank(items, ranking_function) == {0: 0, 1: 1, 2: 2}

    items = ["a", "b", "c"]
    ranking_function = lambda x: ord(x) - ord("c")
    calc_rank(items, ranking_function) == {0: 2, 1: 1, 2: 0}

    args:
        items: List[Any] -- the list of items to rank
        ranking_function: Callable[[Any], float] -- the function to rank the items
    """

    # build a list of tuples, where the first element is the index of the item in the list, and the second element is the item
    items_with_index = list(enumerate(items))

    # sort the list of tuples by the ranking_function
    sorted_items_with_index = sorted(
        items_with_index, key=lambda x: ranking_function(x[1])
    )

    # build a dict where the key is the index of the item in the list, and the value is the rank of the item
    rank = {index: i for i, (index, _) in enumerate(sorted_items_with_index)}

    return rank


def rerank(items, ranking_functions):
    """Returns a reranked list of items. The ranking functions are a list of tuples, where the first element is the weight of the ranking function, and the second element is the ranking function.
        The weights must be positive numbers, and the sum of the weights must be 1.0.

    args:
        items: List[Any] -- the list of items to rank
        ranking_functions: List[Tuple[float, Callable[[Any], float]]] -- the list of ranking functions
    """

    # check if the sum of the weights is 1.0
    if abs(sum(weight for weight, _ in ranking_functions) - 1.0) > 1e-6:
        raise ValueError("The sum of the weights must be 1.0")

    # calculate the rank for each item for each ranking function
    overall_rank = np.zeros(len(items))
    for weight, ranking_function in ranking_functions:
        rank_dict = calc_rank(items, ranking_function)
        rank_array = np.zeros(len(items))
        for k, v in rank_dict.items():
            rank_array[k] = v

        rank_array *= weight
        overall_rank += rank_array

    # sort the items by the weighted rank
    sorted_items = [
        item for _, item in sorted(zip(overall_rank, items), key=lambda x: x[0])
    ]

    return sorted_items


def rerank_weighted_score(items, ranking_functions):
    """
    Returns a reranked list of items. The ranking functions are a list of tuples, where the first element is the weight of the ranking function, and the second element is the ranking function.
    The weights must be positive numbers, and the sum of the weights must be 1.0.
    The scores returned by the ranking functions are multiplied by the weight of the ranking function. The sum of the weighted scores is finally used to rank the items.

    args:
        items: List[Any] -- the list of items to rank
        ranking_functions: List[Tuple[float, Callable[[Any], float]]] -- the list of ranking functions
    """

    # check if the sum of the weights is 1.0
    if abs(sum(weight for weight, _ in ranking_functions) - 1.0) > 1e-6:
        raise ValueError("The sum of the weights must be 1.0")

    # calculate the weighted score for each item for each ranking function
    overall_score = np.zeros(len(items))
    for weight, ranking_function in ranking_functions:
        score = np.array([ranking_function(item) for item in items])
        score *= weight
        overall_score += score

    # sort the items by the weighted score
    sorted_items = [
        item for _, item in sorted(zip(overall_score, items), key=lambda x: x[0])
    ]

    return sorted_items


def rerank_weighted_score_normalized(items, ranking_functions):
    """
    Returns a reranked list of items. The ranking functions are a list of tuples, where the first element is the weight of the ranking function, and the second element is the ranking function.
    The weights must be positive numbers, and the sum of the weights must be 1.0.
    The scores returned by the ranking functions are first normalized, so that the minimum score is 0.0 and the maximum score is 1.0 for each of the ranking functions. The normalized scores are then multiplied by the weight of the ranking function. The sum of the weighted scores is finally used to rank the items.
    args:
        items: List[Any] -- the list of items to rank
        ranking_functions: List[Tuple[float, Callable[[Any], float]]] -- the list of ranking functions
    """

    # check if the sum of the weights is 1.0
    if abs(sum(weight for weight, _ in ranking_functions) - 1.0) > 1e-6:
        raise ValueError("The sum of the weights must be 1.0")

    # calculate the weighted score for each item for each ranking function
    overall_score = np.zeros(len(items))
    for weight, ranking_function in ranking_functions:
        score = np.array([ranking_function(item) for item in items])
        score = (score - score.min()) / (score.max() - score.min())
        score *= weight

        overall_score += score

    # sort the items by the weighted score
    sorted_items = [
        item for _, item in sorted(zip(overall_score, items), key=lambda x: x[0])
    ]

    return sorted_items
