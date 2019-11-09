import numpy as np
import numba as nb
from time import time
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import csc_matrix, random, diags


@nb.njit("void(i8, i8, i4[:], i4[:], i4[:], i4[:], i4[:])")
def csr_matmat_pass1(Am, Bn, Ap, Aj, Bp, Bj, Cp):
    """
    Pass 1 computes CSR row pointer for the matrix product C = A * B

    This is an implementation of the SMMP algorithm:
    "Sparse Matrix Multiplication Package (SMMP)"
    Randolph E. Bank and Craig C. Douglas

    http:#citeseer.ist.psu.edu/445062.html
    http:#www.mgnet.org/~douglas/ccd-codes.html

    :param Am: number of rows in A
    :param Bn: number of columns in B (hence C is Am x Bm)
    :param Ap: A row pointer
    :param Aj: A column indices
    :param Bp: B row pointer
    :param Bj: B column indices
    :param Cp: C row pointer
    :return: Cp by reference
    """
    # method that uses O(n) temp storage
    mask = np.full(Bn, -1, dtype=nb.int32)  # initialize to -1

    Cp[0] = 0
    nnz = 0
    for i in range(Am):
        row_nnz = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                if mask[k] != i:
                    mask[k] = i
                    row_nnz += 1

        next_nnz = nnz + row_nnz

        nnz = next_nnz
        Cp[i + 1] = nnz


@nb.njit("void(i8, i8, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:], i4[:], i4[:], f8[:])")
def csr_matmat_pass2(Am, Bn, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx):
    """
    Pass 2 computes CSR entries for matrix C = A*B using the row pointer Cp[] computed in Pass 1.

    This is an implementation of the SMMP algorithm:
    "Sparse Matrix Multiplication Package (SMMP)"
    Randolph E. Bank and Craig C. Douglas

    http:#citeseer.ist.psu.edu/445062.html
    http:#www.mgnet.org/~douglas/ccd-codes.html

    :param Am: number of rows in A
    :param Bn: number of columns in B (hence C is Am x Bm)
    :param Ap: A row pointer
    :param Aj: A column indices
    :param Ax: A data
    :param Bp: B row pointer
    :param Bj: B column indices
    :param Bx: B data
    :param Cp: C row pointer
    :param Cj: C column indices
    :param Cx: C data
    :return: Cp, Cj and Cx by reference
    """

    _next_ = np.full(Bn, -1, dtype=nb.int32)  # initialize to -1
    sums = np.zeros(Bn, dtype=nb.float64)

    nnz = 0
    Cp[0] = 0

    for i in range(Am):
        head = -2
        length = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                sums[k] += Ax[jj] * Bx[kk]
                if _next_[k] == -1:
                    _next_[k] = head
                    head = k
                    length += 1

        for jj in range(length):
            if sums[head] != 0:
                Cj[nnz] = head
                Cx[nnz] = sums[head]
                nnz += 1

            temp = head
            head = _next_[head]

            _next_[temp] = -1  # clear arrays
            sums[temp] = 0

        Cp[i + 1] = nnz


@nb.njit("void(i8, i8, i4[:], i4[:], i4[:], i4[:], i4[:])")
def csc_matmat_pass1(Am, Bn, Ap, Ai, Bp, Bi, Cp):
    """

    :param Am:
    :param Bn:
    :param Ap:
    :param Ai:
    :param Bp:
    :param Bi:
    :param Cp:
    :return:
    """
    csr_matmat_pass1(Bn, Am, Bp, Bi, Ap, Ai, Cp)


@nb.njit("void(i8, i8, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:], i4[:], i4[:], f8[:])")
def csc_matmat_pass2(Am, Bn, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx):
    """

    :param Am:
    :param Bn:
    :param Ap:
    :param Ai:
    :param Ax:
    :param Bp:
    :param Bi:
    :param Bx:
    :param Cp:
    :param Ci:
    :param Cx:
    :return:
    """
    csr_matmat_pass2(Bn, Am, Bp, Bi, Bx, Ap, Ai, Ax, Cp, Ci, Cx)


def csc_dot(A: csc_matrix, B: csc_matrix):
    """

    :param A:
    :param B:
    :return:
    """
    Cp = np.empty(A.shape[0] + 1, dtype=np.int32)

    csc_matmat_pass1(Am=A.shape[0], Bn=B.shape[1],
                     Ap=A.indptr, Ai=A.indices,
                     Bp=B.indptr, Bi=B.indices, Cp=Cp)
    nnz = Cp[-1]
    Ci = np.empty(nnz, dtype=np.int32)
    Cx = np.empty(nnz, dtype=np.float64)

    csc_matmat_pass2(Am=A.shape[0], Bn=B.shape[1],
                     Ap=A.indptr, Ai=A.indices, Ax=A.data,
                     Bp=B.indptr, Bi=B.indices, Bx=B.data,
                     Cp=Cp, Ci=Ci, Cx=Cx)

    N, M = A.shape[0], B.shape[1]
    C = csc_matrix((Cx, Ci, Cp), shape=(N, M))
    return C


if __name__ == '__main__':

    np.random.seed(0)
    k = 2000
    m = k
    n = k + 1

    A = csc_matrix(random(m, n, density=0.1))  # + diags(np.ones(n))
    B = csc_matrix(random(n, m, density=0.1))  # + diags(np.ones(n))
    x = np.random.random(m)

    # ---------------------------------------------------------------------
    # Scipy
    # ---------------------------------------------------------------------
    t = time()
    C = A * B
    print('Scipy\t', time() - t, 's')

    # ---------------------------------------------------------------------
    # CSparse3
    # ---------------------------------------------------------------------
    t = time()
    C2 = csc_dot(A, B)
    print('This\t', time() - t, 's')

    # ---------------------------------------------------------------------
    # check
    # ---------------------------------------------------------------------

    pass_mult = (C.todense() == C2.todense()).all()

    assert pass_mult
    print('mat mat\t', pass_mult)