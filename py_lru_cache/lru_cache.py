#!/usr/bin/env python3

"""
Just a little bit beautified standard CPython LRU-cache because it is:

- thread-safe;
- supports LRU;
- quite efficient;
- show that I can read at least Python parts of CPython interpreter :)

Link to original CPython code: https://github.com/python/cpython/blob/main/Lib/functools.py

Why not just a simple `from functools import lru_cache`?
Because it shows that for me `lru_cache` is not just a blackbox &
it's always interesting to dive into Python sources.
"""
import logging
import threading
import time
from collections import namedtuple
from threading import RLock
from typing import Callable, Tuple, Mapping, Any, Set, Optional, Type

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

WRAPPER_ASSIGNMENTS = (
    '__module__',
    '__name__',
    '__qualname__',
    '__doc__',
    '__annotate__',
    '__type_params__'
)

WRAPPER_UPDATES = (
    '__dict__',
)


class _HashedSeq(list):
    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(
        args: Tuple[Any, ...],
        kwds: Mapping[str, Any],
        typed: bool,
        kwd_mark=(object(),),
        fasttypes: Optional[Set] = None,
        tuple=tuple, type=type, len=len
):
    # small improvement of `fasttypes` as by default it uses mutable set as a default argument
    if fasttypes is None:
        fasttypes = {int, str}

    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def update_wrapper(
        wrapper: Callable,
        wrapped: Callable,
        assigned: Tuple[str] = WRAPPER_ASSIGNMENTS,
        updated: Tuple[str] = WRAPPER_UPDATES
) -> Callable:
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    wrapper.__wrapped__ = wrapped
    return wrapper


def _lru_cache_wrapper(
        user_function: Callable,
        maxsize: Optional[int],
        typed: bool,
        _CacheInfo: Type[_CacheInfo]
) -> Callable:
    sentinel = object()
    make_key = _make_key

    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3

    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get
    cache_len = cache.__len__
    lock = RLock()
    root = []
    root[:] = [root, root, None, None]

    if maxsize == 0:
        def wrapper(*args, **kwds):
            nonlocal misses
            misses += 1
            result = user_function(*args, **kwds)
            return result
    elif maxsize is None:
        def wrapper(*args, **kwds):
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result
    else:
        def wrapper(*args, **kwds):
            nonlocal root, hits, misses, full
            key = make_key(args, kwds, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = user_function(*args, **kwds)
            with lock:
                if key in cache:
                    pass
                elif full:
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    root[KEY] = root[RESULT] = None
                    del cache[oldkey]
                    cache[key] = oldroot
                else:
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    full = (cache_len() >= maxsize)
            return result

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper


def lru_cache(maxsize: Optional[int] = 128, typed: bool = False) -> Callable:
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(decimal.Decimal("3.0")) and f(3.0) will be treated as
    distinct calls with distinct results. Some types such as str and int may
    be cached separately even when typed is false.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)

    """

    if isinstance(maxsize, int):
        if maxsize < 0:
            maxsize = 0
    elif callable(maxsize) and isinstance(typed, bool):
        user_function, maxsize = maxsize, 128
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)
        wrapper.cache_parameters = lambda: {'maxsize': maxsize, 'typed': typed}
        return update_wrapper(wrapper, user_function)
    elif maxsize is not None:
        raise TypeError('Expected first argument to be an integer, a callable, or None')

    def decorating_function(user_function):
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)
        wrapper.cache_parameters = lambda: {'maxsize': maxsize, 'typed': typed}
        return update_wrapper(wrapper, user_function)

    return decorating_function


def cache(user_function: Callable, /) -> Callable:
    """Simple lightweight unbounded cache.  Sometimes called "memoize"."""
    return lru_cache(maxsize=None)(user_function)


def __demo():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("=== cache demonstration (sequential calls to slow_power) ===")

    @cache
    def slow_power(base: int, exp: int) -> int:
        logging.debug(f"Calculating {base}^{exp} (simulating a time-consuming operation)...")
        time.sleep(5)
        return base ** exp

    val1 = slow_power(2, 10)
    logging.info(f"First call slow_power(2, 10) = {val1}")
    val2 = slow_power(2, 10)
    logging.info(f"Second call slow_power(2, 10) (from cache) = {val2}\n")
    logging.info(f"slow_power cache info {slow_power.cache_info()}")

    logging.info("=== lru_cache demonstration (sequential calls to concat_and_len) ===")

    @lru_cache(maxsize=2, typed=True)
    def concat_and_len(text: str, number: int) -> str:
        logging.debug(f"Building a string from '{text}' and {number}...")
        time.sleep(5)
        return f"{text}{number} (len={len(text) + len(str(number))})"

    val = concat_and_len("test", 123)
    logging.info(f"concat_and_len('test', 123) = {val}")
    val = concat_and_len("test", 123)
    logging.info(f"Second call concat_and_len('test', 123) (from cache) = {val}\n")
    val = concat_and_len("test", 321)
    logging.info(f"concat_and_len('test', 321) = {val}")
    val = concat_and_len("test", 321)
    logging.info(f"Second call concat_and_len('test', 321) (from cache) = {val}")
    val = concat_and_len("ttest", 321)
    logging.info(f"concat_and_len('ttest', 321) = {val}")
    val = concat_and_len("ttest", 321)
    logging.info(f"Second call concat_and_len('ttest', 321) (from cache) = {val}")
    val = concat_and_len("test", 123)
    logging.info(f"Third call concat_and_len('test', 123) = {val}")

    logging.info(f"concat_and_len cache info {concat_and_len.cache_info()}")

    logging.info("=== Demonstration of multithreading and caching ===")

    def worker_slow_power(base: int, exp: int):
        thread_id = threading.get_ident()
        logging.info(f"[Thread {thread_id}] Calling slow_power({base}, {exp})...")
        result = slow_power(base, exp)
        logging.info(f"[Thread {thread_id}] slow_power({base}, {exp}) result = {result}")

    def worker_concat_and_len(text: str, number: int):
        thread_id = threading.get_ident()
        logging.info(f"[Thread {thread_id}] Calling concat_and_len('{text}', {number})...")
        result = concat_and_len(text, number)
        logging.info(f"[Thread {thread_id}] concat_and_len('{text}', {number}) result = {result}")

    threads = []

    inputs_slow_power = [(2, 10), (2, 10), (3, 5), (2, 10), (3, 5), (1, 10), (1, 10), (4, 7), (4, 7), (2, 10), (3, 5)]
    for (b, e) in inputs_slow_power:
        t = threading.Thread(target=worker_slow_power, args=(b, e), name=f"Thr-SP-{b}^{e}")
        threads.append(t)

    inputs_concat_len = [("hello", 7), ("hello", 7), ("world", 777), ("", 42), ("hello", 7), ("world", 777), ("", 42)]
    for (txt, num) in inputs_concat_len:
        t = threading.Thread(target=worker_concat_and_len, args=(txt, num), name=f"Thr-CL-{txt}{num}")
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logging.info(f"slow_power cache info {slow_power.cache_info()}")
    logging.info(f"concat_and_len cache info {concat_and_len.cache_info()}")


if __name__ == "__main__":
    __demo()
