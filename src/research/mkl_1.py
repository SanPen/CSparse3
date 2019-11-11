from ctypes import *
import scipy.sparse as spsp
import numpy as np

# June 2nd 2016 version.

# Load the share library
# mkl = cdll.LoadLibrary("libmkl_rt.so")
# mkl = cdll.LoadLibrary(r"C:\Users\PENVERSA\Apps\ReePy37\Library\bin\mkl_rt.dll")
mkl = cdll.LoadLibrary(r"mkl_rt.dll")


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


def csr_dot_csr_t(A_handle, C_handle, nz=None):
    # Calculate (A.T).dot(A) and put result into C
    #
    # This uses one-based indexing
    #
    # Both C.data and A.data must be in np.float32 type.
    #
    # Number of nonzero elements in C must be greater than
    #     or equal to the size of C.data
    #
    # size of C.indptr must be greater than or equal to
    #     1 + (num rows of A).
    #
    # C_data    = np.zeros((nz), dtype=np.single)
    # C_indices = np.zeros((nz), dtype=np.int32)
    # C_indptr  = np.zeros((m+1),dtype=np.int32)

    (a_pointer, ja_pointer, ia_pointer, A) = A_handle
    (c_pointer, jc_pointer, ic_pointer, C) = C_handle

    trans_pointer = byref(c_char(b'T'))
    sort_pointer = byref(c_int(0))

    m, n = A.shape
    sort_pointer = byref(c_int(0))
    m_pointer = byref(c_int(m))     # Number of rows of matrix A
    n_pointer = byref(c_int(n))     # Number of columns of matrix A
    k_pointer = byref(c_int(n))     # Number of columns of matrix B
    # should be n when trans='T'
    # Otherwise, I guess should be m
    ###
    b_pointer = a_pointer
    jb_pointer = ja_pointer
    ib_pointer = ia_pointer
    ###

    if nz is None:
        # *n # m*m # Number of nonzero elements expected
        # probably can use lower value for sparse
        # matrices.
        nz = n * n

    nzmax_pointer = byref(c_int(nz))
    # length of arrays c and jc. (which are data and
    # indices of csr_matrix). So this is the number of
    # nonzero elements of matrix C
    #
    # This parameter is used only if request=0.
    # The routine stops calculation if the number of
    # elements in the result matrix C exceeds the
    # specified value of nzmax.

    info = c_int(-3)
    info_pointer = byref(info)
    request_pointer_list = [byref(c_int(0)), byref(c_int(1)), byref(c_int(2))]
    return_list = []

    request_pointer = request_pointer_list[0]

    ret = mkl.mkl_scsrmultcsr(trans_pointer, request_pointer, sort_pointer,
                              m_pointer, n_pointer, k_pointer,
                              a_pointer, ja_pointer, ia_pointer,
                              b_pointer, jb_pointer, ib_pointer,
                              c_pointer, jc_pointer, ic_pointer,
                              nzmax_pointer, info_pointer)
    info_val = info.value
    return_list += [(ret, info_val)]
    return return_list


def show_csr_internal(A, indent=4):
    # Print data, indptr, and indices
    # of a scipy csr_matrix A
    name = ['data', 'indptr', 'indices']
    mat = [A.data, A.indptr, A.indices]
    for i in range(3):
        str_print = ' ' * indent + name[i] + ':\n%s' % mat[i]
        str_print = str_print.replace('\n', '\n' + ' ' * indent * 2)
        print(str_print)


def fix_for_scipy(C, A):
    """

    :param C:
    :param A:
    :return:
    """
    n = A.shape[1]
    print("fix n", n)
    nz = C.indptr[n] - 1  # -1 as this is still one based indexing.
    print("fix nz", nz)
    data = C.data[:nz]

    C.indptr[:n+1] -= 1
    indptr = C.indptr[:n+1]
    C.indices[:nz] -= 1
    indices = C.indices[:nz]
    return spsp.csr_matrix((data, indices, indptr), shape=(n, n))


def test():
    np.random.seed(42)

    AA= [[1, 0, 0, 1],
         [1, 0, 1, 0],
         [0, 0, 1, 0]]
    AA = np.random.choice([0, 1], size=(3, 750000), replace=True, p=[0.99, 0.01])
    A_original = spsp.csr_matrix(AA)

    A = A_original.astype(np.float32).tocsc()

    A = spsp.csr_matrix( (A.data, A.indices, A.indptr))
    print("A:")
    show_csr_internal(A)
    print(A.todense())
    A.indptr += 1  # convert to 1-based indexing
    A.indices += 1  # convert to 1-based indexing
    A_ptrs = get_csr_handle(A)

    C = spsp.csr_matrix(np.ones((3, 3)), dtype=np.float32)

    C_ptrs = get_csr_handle(C, clear=True)

    print("C:")
    show_csr_internal(C)
    print("=call mkl function=")
    return_list = csr_dot_csr_t(A_ptrs, C_ptrs)
    print("(ret, info):", return_list)
    print("C after calling mkl:")
    show_csr_internal(C)

    C_fix = fix_for_scipy(C, A)
    print("C_fix for scipy:")
    show_csr_internal(C_fix)
    print(C_fix.todense())

    print("Original C after fixing:")
    show_csr_internal(C)

    print("scipy's (A).dot(A.T)")
    scipy_ans = A_original.dot(A_original.T)

    show_csr_internal(scipy_ans)
    print(scipy_ans.todense())


if __name__ == "__main__":
    test()
