import numpy as np
from time import time
from scipy.sparse import csr_matrix, random, diags
from ctypes import *
import platform

if platform.system() == 'Linux':
    mkl = cdll.LoadLibrary("libmkl_rt.so")

elif platform.system() == 'Windows':
    mkl = cdll.LoadLibrary("mkl_rt.dll")

elif platform.system() == 'Darwin':
    raise Exception("I don't own a Mac...")

else:
    raise Exception('Platform unknown', platform.system())


def get_csr_handle(A, clear=False):

    if clear:
        A.indptr[:] = 0
        A.indices[:] = 0
        A.data[:] = 0

    a_pointer = A.data.ctypes.data_as(POINTER(c_float))

    # Array containing non-zero elements of the matrix A.
    # This corresponds to data array of csr_matrix
    # Its length is equal to #non zero elements in A
    # (Can this be longer than actual #non-zero elements?)
    assert A.data.ctypes.data % 16 == 0  # Check alignment

    ja_pointer = A.indices.ctypes.data_as(POINTER(c_int))
    # Array of column indices of all non-zero elements of A.
    # This corresponds to the indices array of csr_matrix
    assert A.indices.ctypes.data % 16 == 0  # Check alignment

    ia_pointer = A.indptr.ctypes.data_as(POINTER(c_int))

    # Array of length m+1.
    # a[ia[i]:ia[i+1]] is the value of nonzero entries of
    # the ith row of A.
    # ja[ia[i]:ia[i+1]] is the column indices of nonzero
    # entries of the ith row of A
    # This corresponds to the indptr array of csr_matrix
    assert A.indptr.ctypes.data % 16 == 0  # Check alignment

    # A_data_size = A.data.size
    # A_indices_size = A.indices.size
    # A_indptr_size = A.indptr.size

    return a_pointer, ja_pointer, ia_pointer, A


def sparse_csr_mm(A, B):
    """
    Perform C = A x B
    :param A: CSR Sparse matrix
    :param B: CSR Sparse matrix
    :return: A x B CSR Sparse matrix
    """
    m, n = A.shape
    nb, p = B.shape
    assert n == nb
    cm = m
    cn = p
    nz = m * n

    assert A.data.ctypes.data % 16 == 0  # Check alignment
    assert A.indices.ctypes.data % 16 == 0  # Check alignment
    assert A.indptr.ctypes.data % 16 == 0  # Check alignment

    assert B.data.ctypes.data % 16 == 0  # Check alignment
    assert B.indices.ctypes.data % 16 == 0  # Check alignment
    assert B.indptr.ctypes.data % 16 == 0  # Check alignment

    a_pointer = A.data.ctypes.data_as(POINTER(c_float))
    ja_pointer = (A.indices - 1).ctypes.data_as(POINTER(c_int))
    ia_pointer = (A.indptr - 1).ctypes.data_as(POINTER(c_int))

    b_pointer = B.data.ctypes.data_as(POINTER(c_float))
    jb_pointer = (B.indices - 1).ctypes.data_as(POINTER(c_int))
    ib_pointer = (B.indptr - 1).ctypes.data_as(POINTER(c_int))

    trans_pointer = byref(c_char(b'N'))
    sort_pointer = byref(c_int(8))
    m_pointer = byref(c_int(m))     # Number of rows of matrix A
    n_pointer = byref(c_int(n))     # Number of columns of matrix A
    k_pointer = byref(c_int(p))     # Number of columns of matrix B
    nzmax_pointer = byref(c_int(nz))
    info = c_int(-3)
    info_pointer = byref(info)
    return_list = []

    # pass 1: only the values of ic are computed
    ic = np.zeros(cm + 1, dtype=np.int64)
    c_pointer = np.empty(cm + 2).ctypes.data_as(POINTER(c_float))  # data
    jc_pointer = np.empty(cm + 2).ctypes.data_as(POINTER(c_int))   # indices
    ic_pointer = ic.ctypes.data_as(POINTER(c_int))  # indptr

    request_pointer = byref(c_int(1))
    ret = mkl.mkl_dcsrmultcsr(trans_pointer, request_pointer, sort_pointer,
                              m_pointer, n_pointer, k_pointer,
                              a_pointer, ja_pointer, ia_pointer,
                              b_pointer, jb_pointer, ib_pointer,
                              c_pointer, jc_pointer, ic_pointer,
                              nzmax_pointer, info_pointer)

    # pass 2: the values of jc and c are computed
    nnz = ic[cm]
    c = np.empty(nnz, dtype=np.float64)
    jc = np.empty(nnz, dtype=np.int64)
    c_pointer = c.ctypes.data_as(POINTER(c_float))  # data
    jc_pointer = jc.ctypes.data_as(POINTER(c_int))  # indices

    request_pointer = byref(c_int(2))
    ret = mkl.mkl_dcsrmultcsr(trans_pointer, request_pointer, sort_pointer,
                              m_pointer, n_pointer, k_pointer,
                              a_pointer, ja_pointer, ia_pointer,
                              b_pointer, jb_pointer, ib_pointer,
                              c_pointer, jc_pointer, ic_pointer,
                              nzmax_pointer, info_pointer)

    info_val = info.value
    return_list += [(ret, info_val)]

    C = csr_matrix((c,  # data
                    jc,    # indices
                    ic),  # indptr
                   shape=(cm, cn))

    return C


def test_csc_mm():
    np.random.seed(0)
    k = 10
    m, n = k, k

    A = csr_matrix(random(m, n, density=0.01)) + diags(np.ones(n))
    B = csr_matrix(random(m, n, density=0.01)) + diags(np.ones(n))

    t1 = time()
    C1 = A * B
    t_scipy = time() - t1
    print('time in scipy', t_scipy)

    t1 = time()
    C2 = sparse_csr_mm(A, B)
    t_mkl = time() - t1
    print('time in mkl', t_mkl)

    print('data in scipy', C1.data)
    print('data in mkl', C2)


if __name__ == '__main__':

    test_csc_mm()
