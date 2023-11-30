import os
import subprocess
import sys
import numpy as np


def _dummyimport():
    import Cython


try:
    from .sort3 import parallelstringsortstart, parallelradixsortstart, parallelsort
except Exception as e:
    cstring = r"""# distutils: language=c++
# distutils: extra_compile_args=/openmp
# distutils: extra_link_args=/openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=True
# cython: overflowcheck.fold=False
# cython: embedsignature=False
# cython: embedsignature.format=c
# cython: cdivision=True
# cython: cdivision_warnings=False
# cython: cpow=True
# cython: c_api_binop_methods=True
# cython: profile=False
# cython: linetrace=False
# cython: infer_types=False
# cython: language_level=3
# cython: c_string_type=bytes
# cython: c_string_encoding=default
# cython: type_version_tag=True
# cython: unraisable_tracebacks=False
# cython: iterable_coroutine=True
# cython: annotation_typing=True
# cython: emit_code_comments=False
# cython: cpp_locals=False

cimport cython
import numpy as np
cimport numpy as np
import cython
from libcpp.string cimport string
from libcpp.vector cimport vector

ctypedef fused real:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble
    cython.size_t
    cython.Py_ssize_t
    cython.Py_hash_t
    cython.Py_UCS4
    

ctypedef fused real2:
    cython.uchar
    cython.ushort
    cython.uint
    cython.ulong
    cython.ulonglong


cdef extern from "<ppl.h>" namespace "concurrency":
    cdef void parallel_sort[T](T first, T last) nogil
    cdef void parallel_radixsort(vector[cython.ulonglong].iterator, vector[cython.ulonglong].iterator) nogil

cpdef void parallelsort(real[:] a):
    cdef Py_ssize_t paraindex=a.shape[0]
    parallel_sort(&a[0], &a[paraindex])

cdef void parallelsorttext(vector[string] &my_vector2, outarray,Py_ssize_t lena):
    cdef Py_ssize_t i

    cdef vector[string].iterator it1
    cdef vector[string].iterator it2
    it1=my_vector2.begin()
    it2=my_vector2.end()

    parallel_sort(it1,it2)
    for i in range(lena):
        outarray[i]=my_vector2[i]

cpdef void parallelstringsortstart(a, outarray):
    cdef Py_ssize_t i
    cdef Py_ssize_t lena = len(a)
    cdef vector[string] my_vector 
    my_vector.reserve(lena)
    for i in range(lena):
            my_vector.push_back(a[i])
    parallelsorttext(my_vector,outarray,lena)

cdef void parallelradixsort(vector[cython.ulonglong] &my_vector2, real2[:] outarray,Py_ssize_t lena ):
    cdef Py_ssize_t i
    cdef vector[cython.ulonglong].iterator it1
    cdef vector[cython.ulonglong].iterator it2
    it1=my_vector2.begin()
    it2=my_vector2.end()
    parallel_radixsort(it1,it2)
    with nogil:
        for i in range(lena):
            outarray[i]=my_vector2[i]

cpdef void parallelradixsortstart(real2[:] a, real2[:] outarray):
    cdef Py_ssize_t i
    cdef Py_ssize_t lena = len(a)
    cdef vector[cython.ulonglong] my_vector 
    my_vector.reserve(lena)
    with nogil:
        for i in range(lena):
                my_vector.push_back(a[i])
    parallelradixsort(my_vector,outarray,lena)



"""
    pyxfile = f"sort3.pyx"
    pyxfilesetup = f"sort3compiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'sort3', 'sources': ['sort3.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='sort3',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .sort3 import parallelstringsortstart, parallelradixsortstart, parallelsort
    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()

stringtypes = ["S", "a", 'U']


def generate_random_arrays(shape, dtype="float64", low=0, high=1):
    return np.random.uniform(low, high, size=shape).astype(dtype)


def radix_sort(a):
    o = np.zeros_like(a)
    parallelradixsortstart(a, o)
    return o


def parallelsort_sort(a):
    o = a.copy()
    parallelsort(o)
    return o


def parallel_string_sort(a):
    if a.dtype.char == 'U':
        return np.sort(a)
    o = np.zeros_like(a)
    parallelstringsortstart(a, o)
    return o


def sort_all(a):
    try:
        if np.can_cast(a, np.uint64):
            return radix_sort(a)
        if a.dtype.char in stringtypes:
            return parallel_string_sort(a)
        return parallelsort_sort(a)
    except Exception as fe:
        sys.stderr.write(f'{fe} - trying it with NumPy')
        sys.stderr.flush()
        return np.sort(a)
