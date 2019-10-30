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

from sys import stdout
import numpy as np  # this is for compatibility with numpy
import numba as nb
from numba.typed import List
from collections import Iterable
from CSparse.int_functions import ialloc, csc_cumsum_i
from CSparse.float_functions import xalloc, csc_spalloc_f
from CSparse.add import csc_add_ff
from CSparse.multiply import csc_multiply_ff, csc_mat_vec_ff
from CSparse.graph import find_islands


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

         for j in range(len(indptr)):               # for every column, same as range(cols + 1)
            for k in range(indptr[j], indptr[j+1]): # for every entry in the column
                i = indices[k]
                value = data[k]
                print(i, j, value)

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

                pass

            elif isinstance(key[0], int) and isinstance(key[1], slice):

                pass

            elif isinstance(key[0], slice) and isinstance(key[1], int):

                pass

            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                pass

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):

                B = CscMat()
                n, B.indptr, B.indices, B.data = csc_sub_matrix(Am=self.m,
                                                                Anz=self.nz,
                                                                Aindptr=self.indptr,
                                                                Aindices=self.indices,
                                                                Adata=self.data,
                                                                rows=key[0],
                                                                cols=key[1])
                B.nz = n
                B.nzmax = n

                return B

        else:
            raise Exception('The indices must be a tuple (- , -)')

    def __setitem__(self, key, value):
        pass

    def __setslice__(self, i, j, sequence):
        pass

    def __str__(self) -> str:
        """
        To string (dense)
        :return: string
        """
        a = self.todense()
        val = "Matrix[" + ("%d" % self.m) + "][" + ("%d" % self.n) + "]\n"
        rows = self.m
        cols = self.n
        for i in range(0, rows):
            for j in range(0, cols):
                x = a[i, j]
                if x is not None:
                    if x == 0:
                        val += '0' + ' ' * 10
                    else:
                        val += "%6.8f " % x
                else:
                    val += ""
            val += '\n'

        return val

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
                                                                             Aindptr=self.indptr,
                                                                             Aindices=self.indices,
                                                                             Adata=self.data,
                                                                             Bm=other.m,
                                                                             Bn=other.n,
                                                                             Bindptr=other.indptr,
                                                                             Bindices=other.indices,
                                                                             Bdata=other.data)
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

    def dot(self, o) -> "CscMat":
        """
        Dot product
        :param o: CscMat instance
        :return: CscMat instance
        """
        C = CscMat()
        C.m, C.n, C.indptr, C.indices, C.data, C.nzmax = csc_multiply_ff(Am=self.m,
                                                                         An=self.n,
                                                                         Aindptr=self.indptr,
                                                                         Aindices=self.indices,
                                                                         Adata=self.data,
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


@nb.njit("c16[:](i8)")
def cpalloc(n):
    return np.zeros(n, dtype=nb.complex128)


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:])")
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
            if Cx is not None:
                Cx[q] = Ax[p]

    return Cm, Cn, Cp, Ci, Cx


@nb.njit("Tuple((i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:], i4[:], i4[:])")
def csc_sub_matrix(Am, Anz, Aindptr, Aindices, Adata, rows, cols):
    """
    Get SCS arbitrary sub-matrix
    :param A: CSC matrix
    :param rows: row indices to keep
    :param cols: column indices to keep
    :return: CSC sub-matrix (n, new_col_ptr, new_row_ind, new_val)
    """
    n_rows = len(rows)
    n_cols = len(cols)
    found = False
    n = 0
    p = 0
    new_val = xalloc(Anz)
    new_row_ind = ialloc(Anz)
    new_col_ptr = ialloc(Am + 1)

    new_col_ptr[p] = 0

    for j in cols:

        for k in range(Aindptr[j], Aindptr[j + 1]):
            # search row_ind[k] in rows
            found = False
            found_idx = 0
            while not found and found_idx < n_rows:
                if Aindices[k] == rows[found_idx]:
                    found = True
                found_idx += 1

            # store the values if the row was found in rows
            if found:  # if the row index is in the designated rows...
                new_val[n] = Adata[k]  # store the value
                new_row_ind[n] = found_idx - 1  # store the index where the original index was found inside "rows"
                n += 1
        p += 1
        new_col_ptr[p] = n

    new_col_ptr[p] = n

    # B = CscMat()
    # B.data = new_val
    # B.indices = new_row_ind
    # B.indptr = new_col_ptr
    # B.nz = n
    # B.nzmax = n
    return n, new_col_ptr, new_row_ind, new_val


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

