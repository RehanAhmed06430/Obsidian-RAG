---
title: Python Tips
tags: [python, programming, tips]
created: 2026-08-15
---

# Python Tips

## Useful Python Techniques

### List Comprehensions
```python
# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i*j for j in range(5)] for i in range(5)]
```

### Dictionary Comprehensions
```python
word_lengths = {word: len(word) for word in ["hello", "world"]}
```

### f-strings
```python
name, age = "Alice", 30
print(f"{name} is {age} years old")
# Formatting: {value:.2f} for floats, {value:>10} for alignment
```

## Data Science Libraries

### NumPy Basics
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.mean(), arr.std(), arr.max())

# Broadcasting
a = np.array([[1], [2], [3]])  # shape (3, 1)
b = np.array([10, 20, 30])     # shape (3,)
result = a + b  # broadcasts to (3, 3)
```

### Pandas Essentials
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({"name": ["A", "B"], "value": [1, 2]})

# Filter
filtered = df[df["value"] > 1]

# Group and aggregate
grouped = df.groupby("category").agg({"value": ["mean", "sum"]})
```

### Matplotlib Plotting
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Line 1")
plt.scatter(x, y, c=colors, cmap="viridis")
plt.legend()
plt.title("My Plot")
plt.show()
```

## Best Practices

### Virtual Environments
Always use virtual environments:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Type Hints
```python
def process_data(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}
```

### Context Managers
```python
# Using with statement
with open("file.txt", "r") as f:
    content = f.read()

# Custom context manager
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    print(f"Elapsed: {time.time() - start:.2f}s")
```

## ML/DS Python Tips
- Use `%%time` in Jupyter for cell timing
- `joblib` for parallel processing
- `tqdm` for progress bars
- `@lru_cache` for memoization

See also: [[Machine Learning]] for ML-specific Python usage.
