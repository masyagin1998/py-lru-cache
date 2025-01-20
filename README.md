# py-lru-cache

`py-lru-cache` is a lightweight Python library providing efficient caching mechanisms through an LRU (Least Recently Used) cache (in fact just a copy-n-paste of some CPython sources). The package includes:

- `lru_cache`: A decorator for caching function calls with LRU strategy.
- `cache`: A decorator over `lru_cache` one without any size limit.

## Features
- Efficient caching mechanism using an LRU strategy.
- Easy-to-use decorator and class-based caching.
- Configurable and lightweight.
- Thread-safety and size-safety (LRU).

## Installation

Install the package using `poetry` or `pip`:

### Using Poetry
```bash
poetry install
```

## Usage

### Using the `lru_cache` Decorator
```python
from py_lru_cache import lru_cache

@lru_cache(max_size=3)
def expensive_function(x):
    print(f"Calculating {x}")
    return x * x

print(expensive_function(2))  # Calculates and caches the result
print(expensive_function(2))  # Returns cached result
```

### Using the `cache` Class
```python
from py_lru_cache import cache

# Create an LRU cache instance with a maximum size of 2
my_cache = cache(max_size=2)

my_cache["a"] = 1
my_cache["b"] = 2
print(my_cache["a"])  # Access the cached value

my_cache["c"] = 3  # Evicts the least recently used item
print("a" in my_cache)  # False, as "a" has been evicted
```

## Running the Demo

The package includes a demo in the `lru_cache.py` file.
To run the demo, use:
```bash
python py_lru_cache/lru_cache.py
```

## Running Tests

The package includes extensive test coverage in `tests/test_lru_cache.py`.
To run the tests, execute the following command:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Start caching smarter with `py-lru-cache`!
