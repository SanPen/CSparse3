import numpy as np
from time import time
from scipy.sparse import csc_matrix, random, diags
from ctypes import *

mkl = cdll.LoadLibrary("libmkl_rt.so")


def sparse_csc_mm(A, B):
    # from: https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/TpuMdSVSf4E
    # A must be in CSC format, could enforce here with a check

    m = A.shape[0]
    n = A.shape[1]
    Bn = B.shape[1]

    Ax = A.data.ctypes.data_as(POINTER(c_double))
    Ai = A.indices.ctypes.data_as(POINTER(c_int))
    Ap =      A.indptr.ctypes.data_as(POINTER(c_int))
    eindptr = A.indptr.ctypes.data_as(POINTER(c_int))
    eindptr_ = cast(pointer(eindptr), POINTER(c_void_p))
    eindptr_.contents.value += sizeof(eindptr._type_)

    C = np.zeros(m, dtype=np.float64)

    Bx = B.data.ctypes.data_as(POINTER(c_double))
    Cx = C.ctypes.data_as(POINTER(c_double))

    alpha_beta = byref(c_double(1.))
    ldb_ldc = byref(c_int(0))
    no_transpose = byref(c_char(b'N'))
    general_matrix_c_order = c_char_p(b'G**C')

    #  mkl_dcscmm(transa, m, n, k,
    #             alpha, matdescra, val, indx, pntrb, pntre,
    #             b, ldb, beta, c, ldc)
    ret = mkl.mkl_dcscmm(no_transpose, byref(c_int(m)), byref(c_int(Bn)), byref(c_int(n)),
                         alpha_beta, general_matrix_c_order, Ax, Ai, Ap, eindptr,
                         Bx, ldb_ldc, alpha_beta, Cx, ldb_ldc)

    return C


def test_csc_mm():
    np.random.seed(0)
    k = 10
    m, n = k, k

    A = csc_matrix(random(m, n, density=0.01)) + diags(np.ones(n))
    B = csc_matrix(random(m, n, density=0.01)) + diags(np.ones(n))

    t1 = time()
    C1 = A * B
    t_scipy = time() - t1
    print('time in scipy', t_scipy)

    t1 = time()
    C2 = sparse_csc_mm(A, B)
    t_mkl = time() - t1
    print('time in mkl',t_mkl)

    print('data in scipy', C1.data)
    print('data in mkl', C2)


if __name__ == '__main__':

    test_csc_mm()
