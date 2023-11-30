# C++ parallel sorting algorithms through Cython - up to 8 times faster than NumPy 

## pip install cythonfastsort

### Tested against Windows / Python 3.11 / Anaconda

## Cython (and a C/C++ compiler) must be installed to use the optimized Cython implementation.


```python

import sys
import timeit
import numpy as np
from cythonfastsort import generate_random_arrays, sort_all


size = 10000000
arras = [
    (size, "float32", -555555555, 555555555),
    (size, "float64", -100000000000, 100000000000),
    (size, np.uint8, 0, 255),
    (size, np.int8, -120, 120),
    (size, np.int16, -30000, 30000),
    (size, np.int32, -555555555, 555555555),
    (size, np.int64, -(sys.maxsize-1)//2, (sys.maxsize-1)//2),
    (size, np.uint16, 0, 60000),
    (size, np.uint32, 0, 555555555),
    (size, np.uint64, 0, sys.maxsize-1),
]

reps = 3
for a in arras:
    arr = generate_random_arrays(*a)
    seq = generate_random_arrays(size // 10, *a[1:])
    s = """u=sort_all(arr)"""
    u = sort_all(arr)
    t1 = timeit.timeit(s, globals=globals(), number=reps) / reps
    print('c++ ', t1)
    s2 = """q=np.sort(arr)"""
    q = np.sort(arr)
    t2 = timeit.timeit(s2, globals=globals(), number=reps) / reps
    print('np ', t2)
    print(np.all(q == u))

    print("-----------------")

haystack = np.array(
    [
        b"Cumings",
        b"Heikkinen",
        b"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        b"aaa",
        b"bbbb()",
        b"Futrelle",
        b"Allen",
        b"Cumings, Mrs. John Bradley (Florence Briggs Thayer)q",
        b"Braund, Mr. Owen Harris",
        b"Heikkinen, Miss. Laina",
        b"Futrelle, Mrs. Jacques Heath (Lily May Peel)",
        b"Allen, Mr. William Henry",
        b"Braund",
    ],
    dtype="S",
)

arr = np.ascontiguousarray(np.concatenate([haystack for _ in range(200000)]))
reps = 5
s = """u=sort_all(arr)"""
u = sort_all(arr)
t1 = timeit.timeit(s, globals=globals(), number=reps) / reps
s = """u=np.sort(arr)"""
q = np.sort(arr)
t2 = timeit.timeit(s, globals=globals(), number=reps) / reps
print('c++ ', t1)
print('np ', t2)
print(np.all(q == u))

# c++  0.2110750000004676
# np  0.6982908999998472
# True
# -----------------
# c++  0.22871500000110245
# np  0.7115663999999621
# True
# -----------------
# c++  0.0869510999994721
# np  0.27525126666781335
# True
# -----------------
# c++  0.09770773333366378
# np  0.28797130000020843
# True
# -----------------
# c++  0.13066273333364128
# np  0.47454773333326256
# True
# -----------------
# c++  0.17967379999997016
# np  0.6125306666666196
# True
# -----------------
# c++  0.19168066666679806
# np  0.6346349666673632
# True
# -----------------
# c++  0.06744649999988421
# np  0.47243033333264367
# True
# -----------------
# c++  0.09263499999966977
# np  0.6186357666665572
# True
# -----------------
# c++  0.11886440000065097
# np  0.6260395666661983
# True
# -----------------
# c++  0.9570277200000419
# np  1.4337052399998356
# True

```