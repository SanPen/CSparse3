import numpy as np


def pass1(A, B):
    """

    :param A:
    :param B:
    :return:
    """
    an, am = A.shape
    Ap = A.indptr
    Bp = B.indptr
    Bj = B.indices
    Aj = A.indices
    Cp = np.zeros(an + 1, int)
    mask = np.zeros(am, int) - 1
    nnz = 0
    for i in range(an):
        row_nnz = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                if mask[k] != i:
                    mask[k] = i
                    row_nnz += 1
        nnz += row_nnz
        Cp[i + 1] = nnz
    return Cp


def pass2(A, B, Cnnz):
    """

    :param A:
    :param B:
    :param Cnnz:
    :return:
    """
    nrow, ncol = A.shape
    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Bp, Bj, Bx = B.indptr, B.indices, B.data

    next = np.zeros(ncol, int) - 1
    sums = np.zeros(ncol, A.dtype)

    Cp = np.zeros(nrow + 1, int)
    Cj = np.zeros(Cnnz, int)
    Cx = np.zeros(Cnnz, A.dtype)
    nnz = 0
    for i in range(nrow):
        head = -2
        length = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j, v = Aj[jj], Ax[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                sums[k] += v * Bx[kk]
                if next[k] == -1:
                    next[k], head = head, k
                    length += 1
        print(i, sums, next)
        for _ in range(length):
            if sums[head] != 0:
                Cj[nnz], Cx[nnz] = head, sums[head]
                nnz += 1
            temp = head
            head = next[head]
            next[temp], sums[temp] = -1, 0
        Cp[i + 1] = nnz
    return Cp, Cj, Cx


def pass0(A, B):
    """

    :param A:
    :param B:
    :return:
    """
    Cp = pass1(A, B)
    nnz = Cp[-1]
    Cp, Cj, Cx = pass2(A, B, nnz)
    N, M = A.shape[0], B.shape[1]
    C = sparse.csr_matrix((Cx, Cj, Cp), shape=(N, M))
    return C
