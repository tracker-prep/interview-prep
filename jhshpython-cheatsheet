#  Python Competitive Programming Cheat Sheet for my JhSh

## Basics
```python
# Input & Output
n = int(input())
arr = list(map(int, input().split()))
print(*arr)  # unpacked print

# Multiple inputs
a, b, c = map(int, input().split())

# String input
s = input().strip()

# Fast I/O (important for big inputs)
import sys
input = sys.stdin.readline
```

---

## 🔢 Data Types & Operations
```python
# Integer division
x = 7 // 2   # 3
y = 7 / 2    # 3.5

# Modulo
m = 7 % 3    # 1

# Power
p = pow(2, 10)   # 1024
p_mod = pow(2, 10, 1000)  # (2^10) % 1000
```

---

## 📦 Collections
### Lists
```python
arr = [1, 2, 3]
arr.append(4)
arr.pop()     # remove last
arr.sort()
arr.reverse()
arr[::2]      # slicing
```

### Tuples (immutable)
```python
t = (1, 2, 3)
```

### Sets
```python
s = {1, 2, 3}
s.add(4)
s.remove(2)
s.union({5,6})
```

### Dictionaries
```python
d = {"a":1, "b":2}
d["c"] = 3
for k,v in d.items():
    print(k, v)
```

---

## ⚡ Useful Functions
```python
max(arr)
min(arr)
sum(arr)
sorted(arr, reverse=True)
len(arr)
any([True, False])
all([True, True])
```

---

## 🔁 Loops & Comprehensions
```python
# For loop
for i in range(5):
    print(i)

# While loop
while n > 0:
    n -= 1

# List comprehension
squares = [x*x for x in range(1, 6)]

# Dict comprehension
freq = {x: arr.count(x) for x in set(arr)}
```

---

## 🎲 Math & Random
```python
import math, random

math.gcd(12, 18)   # 6
math.lcm(12, 18)   # 36
math.sqrt(25)      # 5.0
math.factorial(5)  # 120
random.randint(1, 10)
```

---

## 📚 Itertools
```python
from itertools import permutations, combinations, accumulate, product

list(permutations([1,2,3]))       # all orderings
list(combinations([1,2,3], 2))    # all pairs
list(accumulate([1,2,3,4]))       # prefix sums
list(product([0,1], repeat=3))    # cartesian product
```

---

## 🧵 Strings
```python
s = "hello"
s.upper()       # "HELLO"
s.lower()       # "hello"
s[::-1]         # reverse
s.split()       # ["hello"]
"-".join(["a","b"])  # "a-b"
s.find("e")     # 1
s.count("l")    # 2
```

---

## 📊 Heap / Priority Queue
```python
import heapq
arr = [3,1,4,1,5]
heapq.heapify(arr)   # min-heap
heapq.heappush(arr, 2)
x = heapq.heappop(arr)
```

---

## 🕰️ Collections (Advanced)
```python
from collections import Counter, defaultdict, deque

Counter("aabbbcc")    # {'a':2,'b':3,'c':2}

d = defaultdict(int)
d["x"] += 1

q = deque([1,2,3])
q.append(4)
q.popleft()
```
