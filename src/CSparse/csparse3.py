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
from collections import Iterable


class TripletsMat:

    def __init__(self, m=0, n=0, nz_max=0, values=None):
        # maximum number of entries
        self.nzmax = max(nz_max, 1)

        # number of rows
        self.m = m

        # number of columns
        self.n = n

        # column pointers (size nzmax)
        self.column = ialloc(nz_max)

        # row indices, size nzmax
        self.rows = ialloc(nz_max)

        # numerical values, size nzmax
        self.data = values if values else xalloc(nz_max)

        # # of entries in triplet matrix, -1 for compressed-col
        self.nz = len(self.data)

    def __getitem__(self, key):

        if isinstance(key, tuple):

            if isinstance(key[0], int) and isinstance(key[1], int):

                return self.try_get(i=key[0], j=key[1])

            elif isinstance(key[0], int) and isinstance(key[1], slice):

                k = 0
                value = [0] * (key[0].stop - key[0].start)
                for j in range(key[0].start, key[0].stop, 1):
                    value[k] = self.try_get(i=key[0], j=j)
                    k += 1
                return value

            elif isinstance(key[0], slice) and isinstance(key[1], int):

                k = 0
                value = [0] * (key[1].stop - key[1].start)
                for i in range(key[1].start, key[1].stop, 1):
                    value[k] = self.try_get(i=i, j=key[1])
                    k += 1
                return value

            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                pass

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):
                pass

        else:
            pass

    def __setitem__(self, key, value):
        """
        Set values in the matrix
        :param key: tuple indicating how to access rows or columns
        :param value: value or array to set in the matrix
        """
        if isinstance(key,  tuple):

            if isinstance(key[0], int) and isinstance(key[1], int):  # [i, j]

                self.insert_or_replace(i1=key[0], j1=key[1], value=value)

            elif isinstance(key[0], int) and isinstance(key[1], slice):   # [i, j:k]

                if key[1].start is None and key[1].stop is None:
                    a = 0
                    b = self.n
                else:
                    a = key[1].start
                    b = key[1].stop

                if isinstance(value, Iterable):
                    k = 0
                    for j in range(a, b, 1):
                        self.insert_or_replace(i1=key[0], j1=j, value=value[k])
                        k += 1
                else:
                    k = 0
                    for j in range(a, b, 1):
                        self.insert_or_replace(i1=key[0], j1=j, value=value)
                        k += 1

            elif isinstance(key[0], slice) and isinstance(key[1], int):   # [i:k, j]

                if key[0].start is None and key[0].stop is None:
                    a = 0
                    b = self.m
                else:
                    a = key[0].start
                    b = key[0].stop

                if isinstance(value, Iterable):
                    k = 0
                    for i in range(a, b, 1):
                        self.insert_or_replace(i1=i, j1=key[1], value=value[k])
                        k += 1
                else:
                    k = 0
                    for i in range(a, b, 1):
                        self.insert_or_replace(i1=i, j1=key[1], value=value)
                        k += 1

            elif isinstance(key[0], slice) and isinstance(key[1], slice):   # [i:l, j:k]

                if key[0].start is None and key[0].stop is None:
                    a = 0
                    b = self.m
                else:
                    a = key[0].start
                    b = key[0].stop

                if key[1].start is None and key[1].stop is None:
                    c = 0
                    d = self.n
                else:
                    c = key[1].start
                    d = key[1].stop

                if isinstance(value, Iterable):
                    k = 0
                    for i in range(a, b, 1):
                        l = 0
                        for j in range(c, d, 1):
                            self.insert_or_replace(i1=i, j1=j, value=value[k, l])
                            l += 1
                        k += 1
                else:
                    for i in range(a, b, 1):
                        for j in range(c, d, 1):
                            self.insert_or_replace(i1=i, j1=j, value=value)

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):   # [i, [d, e, f]]

                if isinstance(value, Iterable):
                    k = 0
                    for j in key[1]:
                        self.insert_or_replace(i1=key[0], j1=j, value=value[k])
                        k += 1
                else:
                    k = 0
                    for j in key[1]:
                        self.insert_or_replace(i1=key[0], j1=j, value=value)
                        k += 1

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):   # [[a, b, c], j]

                if isinstance(value, Iterable):
                    k = 0
                    for i in key[0]:
                        self.insert_or_replace(i1=i, j1=key[1], value=value[k])
                        k += 1
                else:
                    k = 0
                    for i in key[0]:
                        self.insert_or_replace(i1=i, j1=key[1], value=value)
                        k += 1

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):   # [[a, b, c], [d, e, f]]

                if key[0].start is None and key[0].stop is None:
                    a = 0
                    b = self.m
                else:
                    a = key[0].start
                    b = key[0].stop

                if key[1].start is None and key[1].stop is None:
                    c = 0
                    d = self.n
                else:
                    c = key[1].start
                    d = key[1].stop

                if isinstance(value, Iterable):
                    k = 0
                    for i in key[0]:
                        l = 0
                        for j in key[1]:
                            self.insert_or_replace(i1=i, j1=j, value=value[k, l])
                            l += 1
                        k += 1
                else:
                    for i in key[0]:
                        for j in key[1]:
                            self.insert_or_replace(i1=i, j1=j, value=value)

            else:  # other stuff that is not supported
                pass

        else:
            pass  # the key must always be a tuple

    def insert_or_replace(self, i1, j1, value):
        """
        Insert or replace
        :param i1: row index
        :param j1: column index
        :param value: value
        """
        try:
            i = self.rows.index(i1)
            try:
                j = self.column.index(j1)
                if i == j:  # the element was there already
                    self.data[i] = value
                else:
                    self.insert(i=i1, j=j1, value=value)
            except ValueError:
                self.insert(i=i1, j=j1, value=value)
        except ValueError:
            self.insert(i=i1, j=j1, value=value)

    def try_get(self, i, j):

        try:
            i = self.rows.index(i)

            try:
                j = self.column.index(j)

                if i == j:  # the element was there already
                    return self.data[i]
                else:
                    return 0.0

            except ValueError:
                return 0.0

        except ValueError:
            return 0.0

    def __str__(self) -> str:
        """
        String representation
        :return: string
        """
        a = self.to_dense()
        val = "Matrix[" + ("%d" % self.m) + "][" + ("%d" % self.n) + "]\n"
        rows = self.m
        cols = self.n
        for i in range(0, rows):
            for j in range(0, cols):
                x = a[i][j]
                if x is not None:
                    if x == 0:
                        val += '0' + ' ' * 10
                    else:
                        val += "%6.8f " % x
                else:
                    val += ""
            val += '\n'

        return val

    def insert(self, i, j, value):
        """
        Insert triplet
        :param i: row index
        :param j: column index
        :param value: value
        """
        self.rows.append(i)
        self.column.append(j)
        self.data.append(value)
        self.nz = len(self.data)

    def to_csc(self) -> "CscMat":
        """
        Return the CSC version of this matrix
        :return: CscMat matrix
        """
        Cm, Cn, Cp, Ci, Cx = triplets_to_csc(m=self.m, n=self.n, Ti=self.rows, Tj=self.column, Tx=self.data, nz=self.nz)
        mat = CscMat(m=self.m, n=self.n, nz_max=self.nz)
        mat.indices = Ci
        mat.indptr = Cp
        mat.data = Cx
        return mat

    def to_dense(self):
        """
        Pass this matrix to a dense 2D array
        :return: list of lists
        """
        val = [[0 for x in range(self.n)] for y in range(self.m)]
        for i, j, x in zip(self.rows, self.column, self.data):
            val[i][j] = x
        return val


class CscMat:
    """
    Matrix in compressed-column or triplet form.
    """
    def __init__(self, m=0, n=0, nz_max=0):
        """
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

        # column pointers (size n+1) or col indices (size nzmax)
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

    def __add__(self, o) -> "CscMat":
        """
        Matrix addition
        :param o: CscMat instance
        :return: CscMat instance
        """

        if isinstance(o, CscMat):
            C = CscMat()
            C.m, C.n, C.indptr, C.indices, C.data = csc_add(Am=self.m, An=self.n, Aindptr=self.indptr,
                                                            Aindices=self.indices, Adata=self.data,
                                                            Bm=o.m, Bn=o.n, Bindptr=o.indptr,
                                                            Bindices=o.indices, Bdata=o.data,
                                                            alpha=1.0, beta=1.0)
        elif isinstance(o, float) or isinstance(o, int):
            C = self.copy()
            C.data += o
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
            C.m, C.n, C.indptr, C.indices, C.data = csc_add(Am=self.m, An=self.n, Aindptr=self.indptr,
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
            C.m, C.n, C.indptr, C.indices, C.data, C.nzmax = csc_multiply(Am=self.m,
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
            return csc_mat_vec(m=self.m,
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
        C.m, C.n, C.indptr, C.indices, C.data, C.nzmax = csc_multiply(Am=self.m,
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


@nb.njit("i4[:](i8)")
def ialloc(n):
    return np.zeros(n, dtype=nb.int32)


@nb.njit("c16[:](i8)")
def cpalloc(n):
    return np.zeros(n, dtype=nb.complex128)


@nb.njit("f8[:](i8)")
def xalloc(n):
    return np.zeros(n, dtype=nb.float64)


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i8)")
def csc_spalloc(m, n, nzmax):
    """
    Allocate a sparse matrix (triplet form or compressed-column form).

    @param m: number of rows
    @param n: number of columns
    @param nzmax: maximum number of entries
    @return: m, n, Aindptr, Aindices, Adata, Anzmax
    """
    Anzmax = max(nzmax, 1)
    Aindptr = ialloc(n + 1)
    Aindices = ialloc(Anzmax)
    Adata = xalloc(Anzmax)
    return m, n, Aindptr, Aindices, Adata, Anzmax


@nb.njit("i8(i4[:], i4[:], f8[:], i8, f8, i4[:], f8[:], i8, i4[:], i8)")
def csc_scatter(Ap, Ai, Ax, j, beta, w, x, mark, Ci, nz):
    """
    Scatters and sums a sparse vector A(:,j) into a dense vector, x = x + beta * A(:,j)
    :param Ap:
    :param Ai:
    :param Ax:
    :param j: the column of A to use
    :param beta: scalar multiplied by A(:,j)
    :param w: size m, node i is marked if w[i] = mark
    :param x: size m, ignored if null
    :param mark: mark value of w
    :param Ci: pattern of x accumulated in C.i
    :param nz: pattern of x placed in C starting at C.i[nz]
    :return: new value of nz, -1 on error
    """

    for p in range(Ap[j], Ap[j + 1]):
        i = Ai[p]  # A(i,j) is nonzero
        if w[i] < mark:
            w[i] = mark  # i is new entry in column j
            Ci[nz] = i  # add i to pattern of C(:,j)
            nz += 1
            x[i] = beta * Ax[p]  # x(i) = beta*A(i,j)
        else:
            x[i] += beta * Ax[p]  # i exists in C(:,j) already
    return nz


@nb.njit("i8(i4[:], i4[:], i8)")
def csc_cumsum(p, c, n):
    """
    p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c

    @param p: size n+1, cumulative sum of c
    @param c: size n, overwritten with p [0..n-1] on output
    @param n: length of c
    @return: sum (c), null on error
    """
    nz = 0
    nz2 = 0.0
    # if p is None or c is None: return -1 # check inputs
    for i in range(n):
        p[i] = nz
        nz += c[i]
        nz2 += c[i]              # also in double to avoid CS_INT overflow
        c[i] = p[i]             # also copy p[0..n-1] back into c[0..n-1]
    p[n] = nz
    return int(nz2)               # return sum (c [0..n-1])


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:], i8)")
def triplets_to_csc(m, n, Ti, Tj, Tx, nz):
    """
    C = compressed-column form of a triplet matrix T. The columns of C are
    not sorted, and duplicate entries may be present in C.

    @param T: triplet matrix
    @return: Cm, Cn, Cp, Ci, Cx
    """

    Cm, Cn, Cp, Ci, Cx, nz = csc_spalloc(m, n, nz)  # allocate result

    w = ialloc(n)  # get workspace

    for k in range(nz):
        w[Tj[k]] += 1  # column counts

    csc_cumsum(Cp, w, n)  # column pointers

    for k in range(nz):
        p = w[Tj[k]]
        w[Tj[k]] += 1
        Ci[p] = Ti[k]  # A(i,j) is the pth entry in C
        # if Cx is not None:
        Cx[p] = Tx[k]

    return Cm, Cn, Cp, Ci, Cx


@nb.njit("(f8[:], f8[:], i8)")
def _copy_f(src, dest, length):
    for i in range(length):
        dest[i] = src[i]


@nb.njit("(i4[:], i4[:], i8)")
def _copy_i(src, dest, length):
    for i in range(length):
        dest[i] = src[i]


@nb.njit("Tuple((i4[:], f8[:], i8))(i8, i4[:], i4[:], f8[:], i8)")
def csc_sprealloc(An, Aindptr, Aindices, Adata, nzmax):
    """
    Change the max # of entries a sparse matrix can hold.
    :param An:
    :param Aindptr:
    :param Aindices:
    :param Adata:
    :param nzmax:new maximum number of entries
    :return: indices, data, nzmax
    """
    """
    @param A: column-compressed matrix
    @param nzmax:
    @return: true if successful, false on error
    """

    if nzmax <= 0:
        nzmax = Aindptr[An]

    length = min(nzmax, len(Aindices))
    Ainew = ialloc(nzmax)
    _copy_i(Aindices, Ainew, length)

    length = min(nzmax, len(Adata))
    Axnew = xalloc(nzmax)
    _copy_f(Adata, Axnew, length)

    return Ainew, Axnew, nzmax


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:], f8, f8)")
def csc_add(Am, An, Aindptr, Aindices, Adata,
            Bm, Bn, Bindptr, Bindices, Bdata, alpha, beta):
    """
    C = alpha*A + beta*B

    @param A: column-compressed matrix
    @param B: column-compressed matrix
    @param alpha: scalar alpha
    @param beta: scalar beta
    @return: C=alpha*A + beta*B, null on error (Cm, Cn, Cp, Ci, Cx)
    """
    nz = 0

    # if Am != Bm or An != Bn:
    #     return None

    m, anz, n, Bp, Bx = Am, Aindptr[An], Bn, Bindptr, Bdata

    bnz = Bp[n]

    w = ialloc(m)  # get workspace

    x = xalloc(m)   # get workspace

    Cm, Cn, Cp, Ci, Cx, Cnzmax = csc_spalloc(m, n, anz + bnz)  # allocate result

    for j in range(n):
        Cp[j] = nz  # column j of C starts here

        nz = csc_scatter(Aindptr, Aindices, Adata, j, alpha, w, x, j + 1, Ci, nz)  # alpha*A(:,j)

        nz = csc_scatter(Bindptr, Bindices, Bdata, j, beta, w, x, j + 1, Ci, nz)  # beta*B(:,j)

        for p in range(Cp[j], nz):
            Cx[p] = x[Ci[p]]

    Cp[n] = nz  # finalize the last column of C

    return Cm, Cn, Cp, Ci, Cx  # success; free workspace, return C


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:])", parallel=True)
def csc_multiply(Am, An, Aindptr, Aindices, Adata,
                 Bm, Bn, Bindptr, Bindices, Bdata):
    """
    Sparse matrix multiplication, C = A*B
    :param Am:
    :param An:
    :param Aindptr:
    :param Aindices:
    :param Adata:
    :param Bm:
    :param Bn:
    :param Bindptr:
    :param Bindices:
    :param Bdata:
    :return: Cm, Cn, Cp, Ci, Cx, Cnzmax
    """

    nz = 0

    m = Am

    anz = Aindptr[An]

    n, Bp, Bi, Bx = Bn, Bindptr, Bindices, Bdata

    bnz = Bp[n]

    w = np.zeros(n, dtype=nb.int32)  # ialloc(m)  # get workspace

    x = np.zeros(n, dtype=nb.float64)  # xalloc(m)  # get workspace

    Cm, Cn, Cp, Ci, Cx, Cnzmax = csc_spalloc(m, n, anz + bnz)  # allocate result

    for j in range(n):

        if nz + m > Cnzmax:
            Ci, Cx, Cnzmax = csc_sprealloc(Cn, Cp, Ci, Cx, 2 * Cnzmax + m)

        Cp[j] = nz  # column j of C starts here

        for p in range(Bp[j], Bp[j + 1]):
            nz = csc_scatter(Aindptr, Aindices, Adata, Bi[p], Bx[p], w, x, j + 1, Ci, nz)

        for p in range(Cp[j], nz):
            Cx[p] = x[Ci[p]]

    Cp[n] = nz  # finalize the last column of C

    Ci, Cx, Cnzmax = csc_sprealloc(Cn, Cp, Ci, Cx, 0)  # remove extra space from C

    return Cm, Cn, Cp, Ci, Cx, Cnzmax


@nb.njit("f8[:](i8, i8, i4[:], i4[:], f8[:], f8[:])", parallel=True)
def csc_mat_vec(m, n, Ap, Ai, Ax, x):
    """
    Sparse matrix times dense column vector, y = A * x.
    :param m: number of rows
    :param n: number of columns
    :param Ap: pointers
    :param Ai: indices
    :param Ax: data
    :param x: size n, vector x
    :return:size m, vector y
    """
    y = np.zeros(n, dtype=nb.float64)
    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            y[Ai[p]] += Ax[p] * x[j]
    return y


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

    # m, n, Ap, Ai, Ax = A.m, A.n, A.indptr, A.indices, A.data

    Cm, Cn, Cp, Ci, Cx, Cnzmax = csc_spalloc(m=n, n=m, nzmax=Ap[n])  # allocate result

    w = ialloc(m)  # get workspace

    for p in range(Ap[n]):
        w[Ai[p]] += 1  # row counts

    csc_cumsum(Cp, w, m)  # row pointers

    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            q = w[Ai[p]]
            w[Ai[p]] += 1
            Ci[q] = j  # place A(i,j) as entry C(j,i)
            if Cx is not None:
                Cx[q] = Ax[p]

    return Cm, Cn, Cp, Ci, Cx


@nb.njit("f8(i8, i4[:], f8[:])")
def csc_norm(n, Ap, Ax):
    """
    Computes the 1-norm of a sparse matrix = max (sum (abs (A))), largest
    column sum.

    @param A: column-compressed matrix
    @return: the 1-norm if successful, -1 on error
    """
    norm = 0

    for j in range(n):
        s = 0
        for p in range(Ap[j], Ap[j + 1]):
            s += abs(Ax[p])
        norm = max(norm, s)
    return norm


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