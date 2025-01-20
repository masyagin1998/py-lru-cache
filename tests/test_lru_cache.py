import copy
import gc
import pickle
import threading
import time
import unittest
import weakref
from functools import partial
from inspect import Signature
from random import choice
from unittest import mock

from py_lru_cache import cache, lru_cache
from py_lru_cache.lru_cache import _CacheInfo


@lru_cache()
def py_cached_func(x, y):
    return 3 * x + y


class TestLRU(unittest.TestCase):
    cached_func = py_cached_func,

    @lru_cache()
    def cached_meth(self, x, y):
        return 3 * x + y

    @staticmethod
    @lru_cache()
    def cached_staticmeth(x, y):
        return 3 * x + y

    def test_lru(self):
        def orig(x, y):
            return 3 * x + y

        f = lru_cache(maxsize=20)(orig)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(maxsize, 20)
        self.assertEqual(currsize, 0)
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 0)

        domain = range(5)
        for i in range(1000):
            x, y = choice(domain), choice(domain)
            actual = f(x, y)
            expected = orig(x, y)
            self.assertEqual(actual, expected)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertTrue(hits > misses)
        self.assertEqual(hits + misses, 1000)
        self.assertEqual(currsize, 20)

        f.cache_clear()  # test clearing
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 0)
        self.assertEqual(currsize, 0)
        f(x, y)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # Test bypassing the cache
        self.assertIs(f.__wrapped__, orig)
        f.__wrapped__(x, y)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # test size zero (which means "never-cache")
        @lru_cache(0)
        def f():
            nonlocal f_cnt
            f_cnt += 1
            return 20

        self.assertEqual(f.cache_info().maxsize, 0)
        f_cnt = 0
        for i in range(5):
            self.assertEqual(f(), 20)
        self.assertEqual(f_cnt, 5)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 5)
        self.assertEqual(currsize, 0)

        # test size one
        @lru_cache(1)
        def f():
            nonlocal f_cnt
            f_cnt += 1
            return 20

        self.assertEqual(f.cache_info().maxsize, 1)
        f_cnt = 0
        for i in range(5):
            self.assertEqual(f(), 20)
        self.assertEqual(f_cnt, 1)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 4)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # test size two
        @lru_cache(2)
        def f(x):
            nonlocal f_cnt
            f_cnt += 1
            return x * 10

        self.assertEqual(f.cache_info().maxsize, 2)
        f_cnt = 0
        for x in 7, 9, 7, 9, 7, 9, 8, 8, 8, 9, 9, 9, 8, 8, 8, 7:
            #    *  *              *                          *
            self.assertEqual(f(x), x * 10)
        self.assertEqual(f_cnt, 4)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 12)
        self.assertEqual(misses, 4)
        self.assertEqual(currsize, 2)

    def test_lru_no_args(self):
        @lru_cache
        def square(x):
            return x ** 2

        self.assertEqual(list(map(square, [10, 20, 10])),
                         [100, 400, 100])
        self.assertEqual(square.cache_info().hits, 1)
        self.assertEqual(square.cache_info().misses, 2)
        self.assertEqual(square.cache_info().maxsize, 128)
        self.assertEqual(square.cache_info().currsize, 2)

    def test_lru_hash_only_once(self):
        # To protect against weird reentrancy bugs and to improve
        # efficiency when faced with slow __hash__ methods, the
        # LRU cache guarantees that it will only call __hash__
        # only once per use as an argument to the cached function.

        @lru_cache(maxsize=1)
        def f(x, y):
            return x * 3 + y

        # Simulate the integer 5
        mock_int = mock.Mock()
        mock_int.__mul__ = mock.Mock(return_value=15)
        mock_int.__hash__ = mock.Mock(return_value=999)

        # Add to cache:  One use as an argument gives one call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 1)
        self.assertEqual(f.cache_info(), (0, 1, 1, 1))

        # Cache hit: One use as an argument gives one additional call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 2)
        self.assertEqual(f.cache_info(), (1, 1, 1, 1))

        # Cache eviction: No use as an argument gives no additional call
        self.assertEqual(f(6, 2), 20)
        self.assertEqual(mock_int.__hash__.call_count, 2)
        self.assertEqual(f.cache_info(), (1, 2, 1, 1))

        # Cache miss: One use as an argument gives one additional call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 3)
        self.assertEqual(f.cache_info(), (1, 3, 1, 1))

    def test_lru_star_arg_handling(self):
        @lru_cache()
        def f(*args):
            return args

        self.assertEqual(f(1, 2), (1, 2))
        self.assertEqual(f((1, 2)), ((1, 2),))

    def test_lru_type_error(self):
        @lru_cache(maxsize=None)
        def infinite_cache(o):
            pass

        @lru_cache(maxsize=10)
        def limited_cache(o):
            pass

        with self.assertRaises(TypeError):
            infinite_cache([])

        with self.assertRaises(TypeError):
            limited_cache([])

    def test_lru_with_maxsize_none(self):
        @lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)

        self.assertEqual([fib(n) for n in range(16)],
                         [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))

    def test_lru_with_maxsize_negative(self):
        @lru_cache(maxsize=-10)
        def eq(n):
            return n

        for i in (0, 1):
            self.assertEqual([eq(n) for n in range(150)], list(range(150)))
        self.assertEqual(eq.cache_info(), _CacheInfo(hits=0, misses=300, maxsize=0, currsize=0))

    def test_lru_with_exceptions(self):
        for maxsize in (None, 128):
            @lru_cache(maxsize)
            def func(i):
                return 'abc'[i]

            self.assertEqual(func(0), 'a')
            with self.assertRaises(IndexError) as cm:
                func(15)
            self.assertIsNone(cm.exception.__context__)
            # Verify that the previous exception did not result in a cached entry
            with self.assertRaises(IndexError):
                func(15)

    def test_lru_with_types(self):
        for maxsize in (None, 128):
            @lru_cache(maxsize=maxsize, typed=True)
            def square(x):
                return x * x

            self.assertEqual(square(3), 9)
            self.assertEqual(type(square(3)), type(9))
            self.assertEqual(square(3.0), 9.0)
            self.assertEqual(type(square(3.0)), type(9.0))
            self.assertEqual(square(x=3), 9)
            self.assertEqual(type(square(x=3)), type(9))
            self.assertEqual(square(x=3.0), 9.0)
            self.assertEqual(type(square(x=3.0)), type(9.0))
            self.assertEqual(square.cache_info().hits, 4)
            self.assertEqual(square.cache_info().misses, 4)

    def test_lru_cache_typed_is_not_recursive(self):
        cached = lru_cache(typed=True)(repr)

        self.assertEqual(cached(1), '1')
        self.assertEqual(cached(True), 'True')
        self.assertEqual(cached(1.0), '1.0')
        self.assertEqual(cached(0), '0')
        self.assertEqual(cached(False), 'False')
        self.assertEqual(cached(0.0), '0.0')

        self.assertEqual(cached((1,)), '(1,)')
        self.assertEqual(cached((True,)), '(1,)')
        self.assertEqual(cached((1.0,)), '(1,)')
        self.assertEqual(cached((0,)), '(0,)')
        self.assertEqual(cached((False,)), '(0,)')
        self.assertEqual(cached((0.0,)), '(0,)')

        class T(tuple):
            pass

        self.assertEqual(cached(T((1,))), '(1,)')
        self.assertEqual(cached(T((True,))), '(1,)')
        self.assertEqual(cached(T((1.0,))), '(1,)')
        self.assertEqual(cached(T((0,))), '(0,)')
        self.assertEqual(cached(T((False,))), '(0,)')
        self.assertEqual(cached(T((0.0,))), '(0,)')

    def test_lru_with_keyword_args(self):
        @lru_cache()
        def fib(n):
            if n < 2:
                return n
            return fib(n=n - 1) + fib(n=n - 2)

        self.assertEqual(
            [fib(n=number) for number in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        )
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=28, misses=16, maxsize=128, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=0, misses=0, maxsize=128, currsize=0))

    def test_lru_with_keyword_args_maxsize_none(self):
        @lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n=n - 1) + fib(n=n - 2)

        self.assertEqual([fib(n=number) for number in range(16)],
                         [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))

    def test_kwargs_order(self):
        # PEP 468: Preserving Keyword Argument Order
        @lru_cache(maxsize=10)
        def f(**kwargs):
            return list(kwargs.items())

        self.assertEqual(f(a=1, b=2), [('a', 1), ('b', 2)])
        self.assertEqual(f(b=2, a=1), [('b', 2), ('a', 1)])
        self.assertEqual(f.cache_info(), _CacheInfo(hits=0, misses=2, maxsize=10, currsize=2))

    def test_need_for_rlock(self):
        # This will deadlock on an LRU cache that uses a regular lock

        @lru_cache(maxsize=10)
        def test_func(x):
            'Used to demonstrate a reentrant lru_cache call within a single thread'
            return x

        class DoubleEq:
            'Demonstrate a reentrant lru_cache call within a single thread'

            def __init__(self, x):
                self.x = x

            def __hash__(self):
                return self.x

            def __eq__(self, other):
                if self.x == 2:
                    test_func(DoubleEq(1))
                return self.x == other.x

        test_func(DoubleEq(1))  # Load the cache
        test_func(DoubleEq(2))  # Load the cache
        self.assertEqual(test_func(DoubleEq(2)),  # Trigger a re-entrant __eq__ call
                         DoubleEq(2))  # Verify the correct return value

    def test_lru_method(self):
        class X(int):
            f_cnt = 0

            @lru_cache(2)
            def f(self, x):
                self.f_cnt += 1
                return x * 10 + self

        a = X(5)
        b = X(5)
        c = X(7)
        self.assertEqual(X.f.cache_info(), (0, 0, 2, 0))

        for x in 1, 2, 2, 3, 1, 1, 1, 2, 3, 3:
            self.assertEqual(a.f(x), x * 10 + 5)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 0, 0))
        self.assertEqual(X.f.cache_info(), (4, 6, 2, 2))

        for x in 1, 2, 1, 1, 1, 1, 3, 2, 2, 2:
            self.assertEqual(b.f(x), x * 10 + 5)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 4, 0))
        self.assertEqual(X.f.cache_info(), (10, 10, 2, 2))

        for x in 2, 1, 1, 1, 1, 2, 1, 3, 2, 1:
            self.assertEqual(c.f(x), x * 10 + 7)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 4, 5))
        self.assertEqual(X.f.cache_info(), (15, 15, 2, 2))

        self.assertEqual(a.f.cache_info(), X.f.cache_info())
        self.assertEqual(b.f.cache_info(), X.f.cache_info())
        self.assertEqual(c.f.cache_info(), X.f.cache_info())

    def test_pickle(self):
        cls = self.__class__
        for f in cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth:
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(proto=proto, func=f):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    self.assertIs(f_copy, f)

    def test_copy(self):
        cls = self.__class__

        def orig(x, y):
            return 3 * x + y

        part = partial(orig, 2)
        funcs = (cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth, lru_cache(2)(part))
        for f in funcs:
            with self.subTest(func=f):
                f_copy = copy.copy(f)
                self.assertIs(f_copy, f)

    def test_deepcopy(self):
        cls = self.__class__

        def orig(x, y):
            return 3 * x + y

        part = partial(orig, 2)
        funcs = (cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth, lru_cache(2)(part))
        for f in funcs:
            with self.subTest(func=f):
                f_copy = copy.deepcopy(f)
                self.assertIs(f_copy, f)

    def test_lru_cache_parameters(self):
        @lru_cache(maxsize=2)
        def f():
            return 1

        self.assertEqual(f.cache_parameters(), {'maxsize': 2, "typed": False})

        @lru_cache(maxsize=1000, typed=True)
        def f():
            return 1

        self.assertEqual(f.cache_parameters(), {'maxsize': 1000, "typed": True})

    def test_lru_cache_weakrefable(self):
        @lru_cache
        def test_function(x):
            return x

        class A:
            @lru_cache
            def test_method(self, x):
                return (self, x)

            @staticmethod
            @lru_cache
            def test_staticmethod(x):
                return (self, x)

        refs = [weakref.ref(test_function),
                weakref.ref(A.test_method),
                weakref.ref(A.test_staticmethod)]

        for ref in refs:
            self.assertIsNotNone(ref())

        del A
        del test_function
        gc.collect()

        for ref in refs:
            self.assertIsNone(ref())

    def test_common_signatures(self):
        def orig(a, /, b, c=True): ...

        lru = lru_cache(1)(orig)

        self.assertEqual(str(Signature.from_callable(lru)), '(a, /, b, c=True)')
        self.assertEqual(str(Signature.from_callable(lru.cache_info)), '()')
        self.assertEqual(str(Signature.from_callable(lru.cache_clear)), '()')

    def test_lru_cache_threaded(self):
        n, m = 5, 11

        def orig(x, y):
            return 3 * x + y

        @lru_cache(maxsize=n * m)
        def f(x, y):
            return orig(x, y)

        # Initial check
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(currsize, 0)

        # The event that starts all threads
        start_event = threading.Event()

        # Worker that calls f(k, 0) repeatedly
        def full(k):
            start_event.wait()  # block until main says "go"
            for _ in range(m):
                self.assertEqual(f(k, 0), orig(k, 0))

        # A separate worker that clears the cache repeatedly
        def clearer():
            start_event.wait()
            for _ in range(2 * m):
                f.cache_clear()

        # First set: fill the cache with n threads
        threads = [threading.Thread(target=full, args=(k,)) for k in range(n)]
        for t in threads:
            t.start()
        start_event.set()  # let them run
        for t in threads:
            t.join()

        hits, misses, maxsize, currsize = f.cache_info()
        # We loaded n distinct keys, each called m times
        # (the exact hits/misses can vary somewhat by scheduling)
        self.assertEqual(currsize, n)

        # Second set: spawn 1 clearing thread + n filler threads
        start_event.clear()  # reset for next usage
        threads = [threading.Thread(target=clearer)]
        threads += [threading.Thread(target=full, args=(k,)) for k in range(n)]
        for t in threads:
            t.start()
        start_event.set()
        for t in threads:
            t.join()

    def test_lru_cache_threaded2(self):
        n, m = 5, 7

        # Barriers for synchronization
        start_barrier = threading.Barrier(n + 1)
        pause_barrier = threading.Barrier(n + 1)
        stop_barrier = threading.Barrier(n + 1)

        @lru_cache(maxsize=m * n)
        def f(x):
            # Wait on the pause barrier so we all step in unison
            pause_barrier.wait()
            return 3 * x

        self.assertEqual(f.cache_info(), (0, 0, m * n, 0))

        def worker():
            for i in range(m):
                # Wait for main thread
                start_barrier.wait()
                self.assertEqual(f(i), 3 * i)
                stop_barrier.wait()

        # Spawn n worker threads
        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()

        # The main thread does m cycles of waiting/pausing
        for i in range(m):
            start_barrier.wait()  # release workers
            stop_barrier.reset()  # re-arm the barrier for next loop
            pause_barrier.wait()  # let f() proceed
            start_barrier.reset()  # re-arm for next iteration
            stop_barrier.wait()  # wait for them to finish
            pause_barrier.reset()  # re-arm the pause barrier

            # Check cache stats: after i-th iteration,
            # we should have (i+1)*n calls (all misses so far are 0 hits).
            self.assertEqual(f.cache_info(), (0, (i + 1) * n, m * n, i + 1))

        for t in threads:
            t.join()

    def test_lru_cache_threaded3(self):
        @lru_cache(maxsize=2)
        def f(x):
            time.sleep(0.01)
            return 3 * x

        def worker(i, val):
            self.assertEqual(f(val), 3 * val)

        data = [1, 2, 2, 3, 2]
        threads = [threading.Thread(target=worker, args=(i, v))
                   for i, v in enumerate(data)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestCache(unittest.TestCase):
    def test_cache(self):
        @cache
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)

        self.assertEqual([fib(n) for n in range(16)],
                         [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
                         _CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))


if __name__ == '__main__':
    unittest.main()
