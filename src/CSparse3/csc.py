# Copyright (C) 2006-2011, Timothy A. Davis.
# Copyright (C) 2012, Richard Lincoln.
# Copyright (C) 2019, Santiago Peñate Vera.
# http://www.cise.ufl.edu/research/sparse/CSparse
#
# CSparse.py is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# CSparse.py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this Module; if not, write to the Free Software
# Foundation, Inc, 51 Franklin St, Fifth Floor, Boston, MA 02110-1301

"""
THis is the pure python version where the cython code is outlined
CSparse3.py: a Concise Sparse matrix Python package

@author: Timothy A. Davis
@author: Richard Lincoln
@author: Santiago Peñate Vera
"""

import numpy as np  # this is for compatibility with numpy
import numba as nb
from collections.abc import Iterable
from CSparse3.int_functions import ialloc, csc_cumsum_i
from CSparse3.float_functions import xalloc, csc_spalloc_f
from CSparse3.add import csc_add_ff
from CSparse3.multiply import csc_multiply_ff, csc_mat_vec_ff
from CSparse3.graph import find_islands
from CSparse3.conversions import csc_to_csr
from CSparse3.utils import csc_diagonal, csc_diagonal_from_array, stack_4_by_4_ff, dense_to_str


class CscMat:
    """
    Matrix in compressed-column or triplet form.
    """
    def __init__(self, m=0, n=0, nz_max=0):
        """
        CSC sparse matrix

        Format explanation example
             0  1  2
            _________
        0  | 4       |
        1  | 3  9    |
        2  |    7  8 |
        3  | 3     8 |
        4  |    8  9 |
        5  |    4    |

         cols = 3
         rows = 6
                    0  1  2  3  4  5  6  7  8  9   <-- These are the positions indicated by indptr (jut to illustrate)
         data =    [4, 3, 3, 9, 7, 8, 4, 8, 8, 9]      # stores the values
         indices = [0, 1, 3, 1, 2, 4, 5, 2, 3, 4]      # indicates the row index
         indptr  = [0, 3, 7, 10]                       # The length is cols + 1, stores the from and to indices that
                                                         delimit a column.
                                                         i.e. the first column takes the indices and data from the
                                                         positions 0 to 3-1, this is
                                                         column_idx = 0        # (j)
                                                         indices = [0 , 1, 3]  # row indices (i) of the column (j)
                                                         data    = [10, 3, 3]

         Typical loop:

         for j in range(n):  # for every column, same as range(cols)
            for k in range(indptr[j], indptr[j+1]): # for every entry in the column
                i = indices[k]
                value = data[k]
                print(i, j, value)

        For completeness, the CSR equivalent is
                   0  1  2  3  4  5  6  7  8  9
        data =    [4, 3, 9, 7, 8, 3, 8, 8, 9, 4]
        indices = [0, 0, 1, 1, 2, 0, 2, 1, 2, 1]
        indptr =  [0, 1, 3, 5, 7, 9, 10]

        @param m: number of rows
        @param n: number of columns
        @param nz_max: maximum number of entries
        """

        # maximum number of entries
        self.nzmax = max(nz_max, 1)

        # number of rows
        self.m = m

        # number of columns
        self.n = n

        # column pointers (size n+1) or col indices (size n+1)
        # they indicate from which to which index does each column take
        self.indptr = ialloc(n + 1)

        # row indices, size nzmax
        self.indices = ialloc(nz_max)

        # numerical values, size nzmax
        self.data = xalloc(nz_max)

        # -1 for compressed-col
        self.nz = -1

    def __getitem__(self, key):

        if isinstance(key, tuple):

            if isinstance(key[0], int) and isinstance(key[1], int):
                # (a, b) -> value

                pass

            elif isinstance(key[0], int) and isinstance(key[1], slice):
                # (a, :) -> row a

                pass

            elif isinstance(key[0], slice) and isinstance(key[1], int):
                # (:, b) -> column b

                pass

            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                # (:, :) -> self
                return self

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):
                # (a, list_b) -> vector of row a and columns given by list_b
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):
                # (list_a, b) -> vector of column b and rows given by list_a
                pass

            elif isinstance(key[0], slice) and isinstance(key[1], Iterable):
                # (:, list_b) -> Submatrix with the columns given by list_b
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], slice):
                # (list_a, :) -> Submatrix with the rows given by list_a
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):
                # (list_a, list_b)  -> non continous sub-matrix

                B = CscMat()
                n, B.indptr, B.indices, B.data = csc_sub_matrix(Am=self.m,
                                                                Anz=self.nz,
                                                                Ap=self.indptr,
                                                                Ai=self.indices,
                                                                Ax=self.data,
                                                                rows=key[0],
                                                                cols=key[1])
                B.nz = n
                B.nzmax = n

                return B

        else:
            raise Exception('The indices must be a tuple :/')

    def __setitem__(self, key, value):
        raise Exception('Setting values is not allowed in a CSC Matrix, use a Lil Matrix instead and convert it to CSC')

    def __setslice__(self, i, j, sequence):
        raise Exception('Setting values is not allowed in a CSC Matrix, use a Lil Matrix instead and convert it to CSC')

    def __str__(self) -> str:
        """
        To string (dense)
        :return: string
        """
        return dense_to_str(self.todense())

    def __add__(self, other) -> "CscMat":
        """
        Matrix addition
        :param other: CscMat instance
        :return: CscMat instance
        """

        if isinstance(other, CscMat):
            C = CscMat()
            C.m, C.n, C.indptr, C.indices, C.data = csc_add_ff(Am=self.m, An=self.n, Aindptr=self.indptr,
                                                               Aindices=self.indices, Adata=self.data,
                                                               Bm=other.m, Bn=other.n, Bindptr=other.indptr,
                                                               Bindices=other.indices, Bdata=other.data,
                                                               alpha=1.0, beta=1.0)
        elif isinstance(other, float) or isinstance(other, int):
            C = self.copy()
            C.data += other
        else:
            raise Exception('Type not supported')
        return C

    def __sub__(self, o) -> "CscMat":
        """
        Matrix subtraction
        :param o: CscMat instance
        :return: CscMat instance
        """

        if isinstance(o, CscMat):
            C = CscMat()
            C.m, C.n, C.indptr, C.indices, C.data = csc_add_ff(Am=self.m, An=self.n, Aindptr=self.indptr,
                                                               Aindices=self.indices, Adata=self.data,
                                                               Bm=o.m, Bn=o.n, Bindptr=o.indptr,
                                                               Bindices=o.indices, Bdata=o.data,
                                                               alpha=1.0, beta=-1.0)
        elif isinstance(o, float) or isinstance(o, int):
            C = self.copy()
            C.data += o
        else:
            raise Exception('Type not supported')

        return C

    def __mul__(self, other):
        """
        Matrix multiplication
        :param other: CscMat instance
        :return: CscMat instance
        """
        if isinstance(other, CscMat):
            # mat-mat multiplication
            C = CscMat()

            C.m, C.n, C.indptr, C.indices, C.data, C.nzmax = csc_multiply_ff(Am=self.m,
                                                                             An=self.n,
                                                                             Ap=self.indptr,
                                                                             Ai=self.indices,
                                                                             Ax=self.data,
                                                                             Bm=other.m,
                                                                             Bn=other.n,
                                                                             Bp=other.indptr,
                                                                             Bi=other.indices,
                                                                             Bx=other.data)
            return C

        elif isinstance(other, np.ndarray):
            # mat-vec multiplication -> vector
            return csc_mat_vec_ff(m=self.m,
                                  n=self.n,
                                  Ap=self.indptr,
                                  Ai=self.indices,
                                  Ax=self.data,
                                  x=other)

        elif isinstance(other, float) or isinstance(other, int):
            C = self.copy()
            C.data *= other
            return C

        else:
            raise Exception('Type not supported')

    def __neg__(self) -> "CscMat":
        """
        Negative of this matrix
        :return: CscMat instance
        """
        return self.__mul__(-1.0)

    def __eq__(self, other) -> bool:
        """
        Equality check
        :param other: instance of CscMat
        :return: True / False
        """
        if self.shape == other.shape:

            for a, b in zip(self.indices, other.indices):
                if a != b:
                    print('different indices')
                    return False

            for a, b in zip(self.indptr, other.indptr):
                if a != b:
                    print('different indptr')
                    return False

            for a, b in zip(self.data, other.data):
                if a != b:
                    print('different data')
                    return False

            return True
        else:
            return False

    def todense(self) -> "np.array":
        """
        Pass this matrix to a dense 2D array
        :return: list of lists
        """
        return csc_to_dense(m=self.m, n=self.n, indptr=self.indptr, indices=self.indices, data=self.data)

    def to_csr(self):
        """
        Get the CSR representation of this matrix
        :return: Bp, Bi, Bx
        """
        nnz = self.indptr[self.n]
        Bp = np.zeros(self.m + 1, dtype=np.int32)
        Bi = np.empty(nnz, dtype=np.int32)
        Bx = np.empty(nnz, dtype=np.float64)

        csc_to_csr(m=self.m, n=self.n, Ap=self.indptr, Ai=self.indices, Ax=self.data, Bp=Bp, Bi=Bi, Bx=Bx)

        return Bp, Bi, Bx

    def get_nnz(self):
        return self.indptr[self.n]

    def dot(self, o) -> "CscMat":
        """
        Dot product
        :param o: CscMat instance
        :return: CscMat instance
        """
        C = CscMat()
        C.m, C.n, C.indptr, C.indices, C.data, C.nzmax = csc_multiply_ff(Am=self.m,
                                                                         An=self.n,
                                                                         Ap=self.indptr,
                                                                         Ai=self.indices,
                                                                         Ax=self.data,
                                                                         Bm=o.m,
                                                                         Bn=o.n,
                                                                         Bindptr=o.indptr,
                                                                         Bindices=o.indices,
                                                                         Bdata=o.data)
        return C

    def t(self):
        """
        Transpose
        :return:
        """
        C = CscMat()
        C.m, C.n, C.indptr, C.indices, C.data = csc_transpose(m=self.m,
                                                              n=self.n,
                                                              Ap=self.indptr,
                                                              Ai=self.indices,
                                                              Ax=self.data)
        return C

    def islands(self):
        """
        Find islands in the matrix
        :return: list of islands
        """
        islands = find_islands(node_number=self.n, indptr=self.indptr, indices=self.indices)
        return [np.sort(island) for island in islands]

    @property
    def shape(self) -> tuple:
        return self.m, self.n

    def copy(self):
        """
        Deep copy of this object
        :return:
        """
        mat = CscMat()
        mat.m, mat.n = self.m, self.n
        mat.nz = -1
        mat.data = self.data.copy()
        mat.indices = self.indices.copy()
        mat.indptr = self.indptr.copy()
        mat.nzmax = self.nzmax
        return mat


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:])", fastmath=True)
def csc_transpose(m, n, Ap, Ai, Ax):
    """
    Transpose matrix
    :param m: A.m
    :param n: A.n
    :param Ap: A.indptr
    :param Ai: A.indices
    :param Ax: A.data
    :return: Cm, Cn, Cp, Ci, Cx
    """

    """
    Computes the transpose of a sparse matrix, C =A';

    @param A: column-compressed matrix
    @param allocate_values: pattern only if false, both pattern and values otherwise
    @return: C=A', null on error
    """

    Cm, Cn, Cp, Ci, Cx, Cnzmax = csc_spalloc_f(m=n, n=m, nzmax=Ap[n])  # allocate result

    w = ialloc(m)  # get workspace

    for p in range(Ap[n]):
        w[Ai[p]] += 1  # row counts

    csc_cumsum_i(Cp, w, m)  # row pointers

    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            q = w[Ai[p]]
            w[Ai[p]] += 1
            Ci[q] = j  # place A(i,j) as entry C(j,i)
            Cx[q] = Ax[p]

    return Cm, Cn, Cp, Ci, Cx


@nb.njit("Tuple((i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:], i4[:], i4[:])")
def csc_sub_matrix(Am, Anz, Ap, Ai, Ax, rows, cols):
    """
    Get SCS arbitrary sub-matrix
    :param A: CSC matrix
    :param rows: row indices to keep
    :param cols: column indices to keep
    :return: CSC sub-matrix (n, new_col_ptr, new_row_ind, new_val)
    """
    n_rows = len(rows)
    n = 0
    p = 0
    Bx = xalloc(Anz)
    Bi = ialloc(Anz)
    Bp = ialloc(Am + 1)

    Bp[p] = 0

    for j in cols:  # for each column selected ...
        for k in range(Ap[j], Ap[j + 1]):  # for each row of the column j of A...
            # search row_ind[k] in rows
            found = False
            found_idx = 0
            while not found and found_idx < n_rows:
                if Ai[k] == rows[found_idx]:
                    found = True
                found_idx += 1

            # store the values if the row was found in rows
            if found:  # if the row index is in the designated rows...
                Bx[n] = Ax[k]  # store the value
                Bi[n] = found_idx - 1  # store the index where the original index was found inside "rows"
                n += 1
        p += 1
        Bp[p] = n

    Bp[p] = n

    return n, Bp, Bi, Bx


@nb.njit("f8[:, :](i8, i8, i4[:], i4[:], f8[:])")
def csc_to_dense(m, n, indptr, indices, data):
    """
    Convert csc matrix to dense
    :param m:
    :param n:
    :param indptr:
    :param indices:
    :param data:
    :return: 2d numpy array
    """
    val = np.zeros((m, n), dtype=nb.float64)

    for j in range(n):
        for p in range(indptr[j], indptr[j + 1]):
            val[indices[p], j] = data[p]
    return val


def scipy_to_mat(scipy_mat):

    mat = CscMat()

    mat.m, mat.n = scipy_mat.shape
    mat.nz = -1
    mat.data = scipy_mat.data
    mat.indices = scipy_mat.indices  # .astype(np.int64)
    mat.indptr = scipy_mat.indptr  # .astype(np.int64)
    mat.nzmax = scipy_mat.nnz

    scipy_mat.tocsr()

    return mat


def Diag(m, n, value=1.0) -> CscMat:
    """
    Convert this matrix into a diagonal matrix with the value in the diagonal
    :param m:
    :param n:
    :param value: float value
    :return CscMat
    """
    A = CscMat(m, n)
    A.indices, A.indptr, A.data = csc_diagonal(A.m, value)
    A.n = A.m
    A.nz = A.indptr[A.n]

    return A


def Diags(array: np.ndarray) -> CscMat:
    """
    Convert array into diagonal matrix
    :param array: numpy array float64
    :return CscMat
    """
    m = array.shape[0]
    n = m
    A = CscMat(m, n)
    A.indices, A.indptr, A.data = csc_diagonal_from_array(A.m, array)
    A.n = A.m
    A.nz = A.indptr[A.n]

    return A


def pack_4_by_4(A11: CscMat, A12: CscMat, A21: CscMat, A22: CscMat):
    """

    :param A11:
    :param A12:
    :param A21:
    :param A22:
    :return:
    """

    m, n, Pi, Pp, Px = stack_4_by_4_ff(am=A11.m, an=A11.n, Ai=A11.indices, Ap=A11.indptr, Ax=A11.data,
                                       bm=A12.m, bn=A12.n, Bi=A12.indices, Bp=A12.indptr, Bx=A12.data,
                                       cm=A21.m, cn=A21.n, Ci=A21.indices, Cp=A21.indptr, Cx=A21.data,
                                       dm=A22.m, dn=A22.n, Di=A22.indices, Dp=A22.indptr, Dx=A22.data)
    P = CscMat(m, n)
    P.indptr = Pp
    P.indices = Pi
    P.data = Px
    return P