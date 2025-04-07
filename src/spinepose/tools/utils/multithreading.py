import concurrent.futures
from typing import Any, Callable, Iterable, List, Optional


def concurrent_forloop(
    func: Callable[..., Any],
    iterable: Iterable,
    *iterables: Iterable,
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Run a function concurrently on an iterable and return the results in order.

    If max_workers is None, the ThreadPoolExecutor will determine the optimal
    number of workers based on system resources.

    This function also supports passing additional iterables to provide varying arguments
    to the function. When additional iterables are provided, the function is called
    with one item from each iterable concurrently, similar to the built-in map function.

    Args:
        func (Callable): The function to run on each element of the iterable.
        iterable (Iterable): The primary iterable to process.
        *iterables (Iterable): Optional additional iterables that provide varying arguments.
        max_workers (Optional[int]): The maximum number of workers to use. Defaults to None,
                                     which lets ThreadPoolExecutor decide based on system
                                     resources.

    Returns:
        List[Any]: A list of results from the function.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, iterable, *iterables))
