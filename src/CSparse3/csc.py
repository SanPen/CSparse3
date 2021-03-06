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

import numpy as np
from collections.abc import Iterable
from CSparse3.utils import dense_to_str
from CSparse3 import __config__
import scipy.sparse.sparsetools as sptools
if __config__.NATIVE:
    try:
        from CSparse3.csc_native import *
        print('Using native code')
    except:
        from CSparse3.csc_numba import *
else:
    from CSparse3.csc_numba import *


class CscMat:
    """
    Matrix in compressed-column or triplet form.
    """
    def __init__(self, m=0, n=0, nz_max=0, indptr=None, indices=None, data=None, zeros=False):
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
            ---------
         cols = 3
         rows = 6
                    0  1  2  3  4  5  6  7  8  9   <-- These are the positions indicated by indptr (just to illustrate)
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

        # number of rows
        self.m = m

        # number of columns
        self.n = n

        if indptr is None:
            # maximum number of entries
            self.nzmax = max(nz_max, 1)

            if zeros:
                # column pointers (size n+1) or col indices (size n+1)
                # they indicate from which to which index does each column take
                self.indptr = np.zeros(n + 1, dtype=np.int32)

                # row indices, size nzmax
                self.indices = np.zeros(nz_max, dtype=np.int32)

                # numerical values, size nzmax
                self.data = np.zeros(nz_max, dtype=np.float64)
            else:
                # column pointers (size n+1) or col indices (size n+1)
                # they indicate from which to which index does each column take
                self.indptr = np.empty(n + 1, dtype=np.int32)

                # row indices, size nzmax
                self.indices = np.empty(nz_max, dtype=np.int32)

                # numerical values, size nzmax
                self.data = np.empty(nz_max, dtype=np.float64)

        else:

            # column pointers (size n+1) or col indices (size n+1)
            # they indicate from which to which index does each column take
            self.indptr = indptr

            # row indices, size nzmax
            self.indices = indices

            # numerical values, size nzmax
            self.data = data

            # maximum number of entries
            self.nzmax = len(self.data)

        # -1 for compressed-col
        # self.nz = -1

    def __getitem__(self, key):

        if isinstance(key, tuple):

            if isinstance(key[0], int) and isinstance(key[1], int):
                # (a, b) -> value

                raise NotImplementedError('Single value extraction not implemented')

            elif isinstance(key[0], int) and isinstance(key[1], slice):
                # (a, :) -> row a
                rows = np.array([key[0]])
                nzmax, Bp, Bi, Bx = csc_sub_matrix_rows(An=self.n,
                                                        Anz=self.nzmax,
                                                        Ap=self.indptr,
                                                        Ai=self.indices,
                                                        Ax=self.data,
                                                        rows=rows)
                m = 1
                n = self.n
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

            elif isinstance(key[0], slice) and isinstance(key[1], int):
                # (:, b) -> column b
                cols = np.array([key[1]])
                nzmax, Bp, Bi, Bx = csc_sub_matrix_cols(self.m,
                                                        self.nzmax,
                                                        self.indptr,
                                                        self.indices,
                                                        self.data,
                                                        cols)
                m = self.m
                n = 1
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                # (:, :) -> self
                return self

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):
                # (a, list_b) -> vector of row a and columns given by list_b

                rows = np.array([key[0]])

                if not isinstance(key[1], np.ndarray):
                    cols = np.array(key[1])
                else:
                    cols = key[1]

                nzmax, Bp, Bi, Bx = csc_sub_matrix(self.m,
                                                   self.nzmax,
                                                   self.indptr,
                                                   self.indices,
                                                   self.data,
                                                   rows,
                                                   cols)
                m = 1
                n = len(key[1])
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):
                # (list_a, b) -> vector of column b and rows given by list_a
                if not isinstance(key[0], np.ndarray):
                    rows = np.array(key[0])
                else:
                    rows = key[0]

                cols = np.array([key[1]])

                nzmax, Bp, Bi, Bx = csc_sub_matrix(self.m,
                                                   self.nzmax,
                                                   self.indptr,
                                                   self.indices,
                                                   self.data,
                                                   rows,
                                                   cols)
                m = len(key[0])
                n = 1
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

            elif isinstance(key[0], slice) and isinstance(key[1], Iterable):
                # (:, list_b) -> Submatrix with the columns given by list_b

                if not isinstance(key[1], np.ndarray):
                    cols = np.array(key[1])
                else:
                    cols = key[1]

                nzmax, Bp, Bi, Bx = csc_sub_matrix_cols(self.m,
                                                        self.nzmax,
                                                        self.indptr,
                                                        self.indices,
                                                        self.data,
                                                        cols)

                m = self.m
                n = len(key[1])
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

            elif isinstance(key[0], Iterable) and isinstance(key[1], slice):
                # (list_a, :) -> Submatrix with the rows given by list_a

                if not isinstance(key[0], np.ndarray):
                    rows = np.array(key[0])
                else:
                    rows = key[0]

                nzmax, Bp, Bi, Bx = csc_sub_matrix_rows(An=self.n,
                                                        Anz=self.nzmax,
                                                        Ap=self.indptr,
                                                        Ai=self.indices,
                                                        Ax=self.data,
                                                        rows=rows)

                m = len(key[0])
                n = self.n
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):
                # (list_a, list_b)  -> non continuous sub-matrix

                if not isinstance(key[0], np.ndarray):
                    rows = np.array(key[0])
                else:
                    rows = key[0]

                if not isinstance(key[1], np.ndarray):
                    cols = np.array(key[1])
                else:
                    cols = key[1]

                nzmax, Bp, Bi, Bx = csc_sub_matrix(self.m,
                                                   self.nzmax,
                                                   self.indptr,
                                                   self.indices,
                                                   self.data,
                                                   rows,
                                                   cols)
                m = len(key[0])
                n = len(key[1])
                return CscMat(m=m, n=n, indptr=Bp, indices=Bi, data=Bx)

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

        if isinstance(other, CscMat):  # matrix-matrix addition
            assert(other.m == self.m)
            assert (other.n == self.n)
            C = CscMat(m=self.m, n=self.n, nz_max=self.nzmax + other.nzmax, zeros=True)
            sptools.csc_plus_csc(self.m, self.n,
                                 self.indptr, self.indices, self.data,
                                 other.indptr, other.indices, other.data,
                                 C.indptr, C.indices, C.data)
            return C

        elif isinstance(other, float) or isinstance(other, int):
            # C = self.copy()
            # C.data += other
            raise NotImplementedError('Adding a nonzero scalar to a sparse matrix would make it a dense matrix.')
        else:
            raise NotImplementedError('Type not supported')

    def __sub__(self, other) -> "CscMat":
        """
        Matrix subtraction
        :param other: CscMat instance
        :return: CscMat instance
        """

        if isinstance(other, CscMat):  # subtract CSC matrix
            assert (other.m == self.m)
            assert (other.n == self.n)
            C = CscMat(m=self.m, n=self.n, nz_max=self.nzmax + other.nzmax, zeros=True)
            sptools.csc_minus_csc(self.m, self.n,
                                  self.indptr, self.indices, self.data,
                                  other.indptr, other.indices, other.data,
                                  C.indptr, C.indices, C.data)
            return C

        elif isinstance(other, float) or isinstance(other, int):  # Add scalar value

            raise NotImplementedError('Adding a non-zero scalar to a sparse matrix would make it a dense matrix.')
        else:
            raise NotImplementedError('Type not supported')

    def __mul__(self, other):
        """
        Matrix multiplication
        :param other: CscMat instance
        :return: CscMat instance
        """
        if isinstance(other, CscMat):  # mat-mat multiplication
            # 2-pass matrix multiplication
            Cp = np.empty(self.n + 1, dtype=np.int32)

            sptools.csc_matmat_pass1(self.n, other.m,
                                     self.indptr, self.indices,
                                     other.indptr, other.indices, Cp)
            nnz = Cp[-1]
            Ci = np.empty(nnz, dtype=np.int32)
            Cx = np.empty(nnz, dtype=np.float64)

            sptools.csc_matmat_pass2(self.n, other.m,
                                     self.indptr, self.indices, self.data,
                                     other.indptr, other.indices, other.data,
                                     Cp, Ci, Cx)

            return CscMat(n=self.m, m=other.m, indptr=Cp, indices=Ci, data=Cx)

        elif isinstance(other, np.ndarray):  # multiply by a vector or array of vectors

            if len(other.shape) == 1:
                y = np.zeros(self.m, dtype=np.float64)
                sptools.csc_matvec(self.m, self.n,
                                   self.indptr, self.indices, self.data,
                                   other, y)
                return y
            elif len(other.shape) == 2:

                '''
                
                 * Input Arguments:
                 *   I  n_row            - number of rows in A
                 *   I  n_col            - number of columns in A
                 *   I  n_vecs           - number of column vectors in X and Y
                 *   I  Ap[n_row+1]      - row pointer
                 *   I  Aj[nnz(A)]       - column indices
                 *   T  Ax[nnz(A)]       - nonzeros
                 *   T  Xx[n_col,n_vecs] - input vector
                 *
                 * Output Arguments:
                 *   T  Yx[n_row,n_vecs] - output vector
                 *
                 * Note:
                 *   Output array Yx must be preallocated
                 *
                
                void csc_matvecs(const I n_row,
                                 const I n_col,
                                 const I n_vecs,
                                 const I Ap[],
                                 const I Ai[],
                                 const T Ax[],
                                 const T Xx[],
                                       T Yx[])
                '''
                n_col, n_vecs = other.shape

                y = np.zeros((self.m, n_vecs), dtype=np.float64)
                sptools.csc_matvecs(self.m, self.n, n_vecs,
                                    self.indptr, self.indices, self.data,
                                    other, y)
                return y

        elif isinstance(other, float) or isinstance(other, int):  # multiply by a scalar value
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
        return csc_to_dense(self.m, self.n, self.indptr, self.indices, self.data)

    def to_csr(self):
        """
        Get the CSR representation of this matrix
        :return: Bp, Bi, Bx
        """
        nnz = self.indptr[self.n]
        Bp = np.zeros(self.m + 1, dtype=np.int32)
        Bi = np.empty(nnz, dtype=np.int32)
        Bx = np.empty(nnz, dtype=np.float64)

        csc_to_csr(self.m, self.n, self.indptr, self.indices, self.data, Bp, Bi, Bx)

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
        C.m, C.n, C.indptr, C.indices, C.data, C.nzmax = csc_multiply_ff(self.m,
                                                                         self.n,
                                                                         self.indptr,
                                                                         self.indices,
                                                                         self.data,
                                                                         o.m,
                                                                         o.n,
                                                                         o.indptr,
                                                                         o.indices,
                                                                         o.data)
        return C

    def t(self):
        """
        Transpose
        :return:
        """
        C = CscMat()
        C.m, C.n, C.indptr, C.indices, C.data = csc_transpose(self.m,
                                                              self.n,
                                                              self.indptr,
                                                              self.indices,
                                                              self.data)
        return C

    def islands(self):
        """
        Find islands in the matrix
        :return: list of islands
        """
        islands = find_islands(self.n, self.indptr, self.indices)
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
        mat.data = self.data.copy()
        mat.indices = self.indices.copy()
        mat.indptr = self.indptr.copy()
        mat.nzmax = self.nzmax
        return mat


def scipy_to_mat(scipy_mat):

    mat = CscMat()

    mat.m, mat.n = scipy_mat.shape
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
    A.nzmax = A.indptr[A.n]

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
    A.nzmax = A.indptr[A.n]

    return A


def pack_4_by_4(A11: CscMat, A12: CscMat, A21: CscMat, A22: CscMat):
    """

    :param A11:
    :param A12:
    :param A21:
    :param A22:
    :return:
    """

    m, n, Pi, Pp, Px = csc_stack_4_by_4_ff(A11.m, A11.n, A11.indices, A11.indptr, A11.data,
                                           A12.m, A12.n, A12.indices, A12.indptr, A12.data,
                                           A21.m, A21.n, A21.indices, A21.indptr, A21.data,
                                           A22.m, A22.n, A22.indices, A22.indptr, A22.data)
    P = CscMat(m, n)
    P.indptr = Pp
    P.indices = Pi
    P.data = Px
    return P