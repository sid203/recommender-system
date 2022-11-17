from functools import reduce
from typing import Callable, List, Any, Tuple, Dict, Generator


def compose(*functions: Callable) -> Callable:
    """
    Composes pipeline of functions together
    :param functions: list of functions
    :return: composed function
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def compose_with_args(functions: List[Tuple[Callable, Any]], init: Any) -> Any:
    """
    Composes pipeline of functions with arguments together
    :param functions: list of tuple of functions with arguments
    :return: composed function
    """
    func_with_args = lambda x, f, a: f(x, *a)

    return reduce(lambda store, func: func_with_args(store, *func), functions, init)


def chunks(lst: List, n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def merge_dicts(dicts: List[Dict]) -> Dict:
    return reduce(lambda x, y: {**x, **y}, dicts)
