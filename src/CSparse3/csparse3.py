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
CSparse.py: a Concise Sparse matrix Python package

@author: Timothy A. Davis
@author: Richard Lincoln
@author: Santiago Peñate Vera
"""

from sys import stdout
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
        return cs_compress(self)

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
    def __init__(self, m=0, n=0, nz_max=0, values=None, triplet=False):
        """
        @param m: number of rows
        @param n: number of columns
        @param nz_max: maximum number of entries
        @param values: allocate pattern only if false, values and pattern otherwise
        @param triplet: compressed-column if false, triplet form otherwise
        """

        # maximum number of entries
        self.nzmax = max(nz_max, 1)

        # number of rows
        self.m = m

        # number of columns
        self.n = n

        # column pointers (size n+1) or col indices (size nzmax)
        self.indptr = ialloc(nz_max) if triplet else ialloc(n + 1)

        # row indices, size nzmax
        self.indices = ialloc(nz_max)

        # numerical values, size nzmax
        self.data = xalloc(nz_max) if values else None

        # # of entries in triplet matrix, -1 for compressed-col
        self.nz = 0 if triplet else -1  # allocate triplet or comp.col

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __setslice__(self, i, j, sequence):
        pass

    def __str__(self) -> str:
        """
        To string (dense)
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

    def __add__(self, other) -> "CscMat":
        """
        Matrix addition
        :param other: CscMat instance
        :return: CscMat instance
        """
        return cs_add(self, other, 1.0, 1.0)

    def __radd__(self, other) -> "CscMat":
        pass

    def __iadd__(self, other) -> "CscMat":
        pass

    def __sub__(self, other) -> "CscMat":
        """
        Matrix subtraction
        :param other: CscMat instance
        :return: CscMat instance
        """
        return cs_add(self, other, 1.0, -1.0)

    def __rsub__(self, other) -> "CscMat":
        pass

    def __isub__(self, other) -> "CscMat":
        pass

    def __mul__(self, other) -> "CscMat":
        """
        Matrix multiplication
        :param other: CscMat instance
        :return: CscMat instance
        """
        if isinstance(other, CscMat):
            return cs_multiply(self, other)
        else:
            mat = self.copy()

            for i in range(len(mat.data)):
                mat.data[i] *= other

            return mat

    def __rmul__(self, other) -> "CscMat":
        pass

    def __imul__(self, other) -> "CscMat":
        pass

    def __truediv__(self, other) -> "CscMat":
        pass

    def __itruediv__(self, other) -> "CscMat":
        pass

    def __neg__(self) -> "CscMat":
        return self.__mul__(-1)

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

    def __le__(self, other) -> "CscMat":
        pass

    def __ge__(self, other) -> "CscMat":
        pass

    def to_dense(self):
        """
        Pass this matrix to a dense 2D array
        :return: list of lists
        """
        val = [[0 for x in range(self.n)] for y in range(self.m)]
        for j in range(self.n):
            for p in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[p]
                val[i][j] = self.data[p]
        return val

    def dot(self, other) -> "CscMat":
        """
        Dot product
        :param other:
        :return:
        """
        return cs_multiply(self, other)

    def t(self):
        return cs_transpose(self)

    @property
    def shape(self):
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


def CS_CSC(A: CscMat):
    """
    Returns true if A is in column-compressed form, false otherwise.

    @param A: sparse matrix
    @return: true if A is in column-compressed form, false otherwise
    """
    return A is not None and A.nz == -1


def CS_TRIPLET(A: CscMat):
    """
    Returns true if A is in triplet form, false otherwise.

    @param A: sparse matrix
    @return: true if A is in triplet form, false otherwise
    """
    return A is not None and A.nz >= 0


def CS_FLIP(i):
    return -i - 2


def CS_UNFLIP(i):
    return CS_FLIP(i) if i < 0 else i


def CS_MARKED(w, j):
    return w[j] < 0


def CS_MARK(w, j):
    w[j] = CS_FLIP(w[j])


def ialloc(n):
    return [0] * n


def xalloc(n):
    return [0.0] * n


def cs_spalloc(m, n, nzmax, values, triplet):
    """
    Allocate a sparse matrix (triplet form or compressed-column form).

    @param m: number of rows
    @param n: number of columns
    @param nzmax: maximum number of entries
    @param values: allocate pattern only if false, values and pattern otherwise
    @param triplet: compressed-column if false, triplet form otherwise
    @return: sparse matrix
    """
    A = CscMat()  # allocate the CscMat object
    A.m = m  # define dimensions and nzmax
    A.n = n
    A.nzmax = nzmax = max(nzmax, 1)
    A.nz = 0 if triplet else -1  # allocate triplet or comp.col
    A.indptr = ialloc(nzmax) if triplet else ialloc(n + 1)
    A.indices = ialloc(nzmax)
    A.data = xalloc(nzmax) if values else None
    return A


def cs_scatter(A: CscMat, j, beta, w, x, mark, C: CscMat, nz):
    """
    Scatters and sums a sparse vector A(:,j) into a dense vector, x = x +
    beta * A(:,j).

    @param A: the sparse vector is A(:,j)
    @param j: the column of A to use
    @param beta: scalar multiplied by A(:,j)
    @param w: size m, node i is marked if w[i] = mark
    @param x: size m, ignored if null
    @param mark: mark value of w
    @param C: pattern of x accumulated in C.i
    @param nz: pattern of x placed in C starting at C.i[nz]
    @return: new value of nz, -1 on error
    """
    # if not CS_CSC(A) or w is None or not CS_CSC(C):
    #     return -1  # check inputs
    Ap, Ai, Ax = A.indptr, A.indices, A.data
    Ci = C.indices
    for p in range(Ap[j], Ap[j + 1]):
        i = Ai[p]  # A(i,j) is nonzero
        if w[i] < mark:
            w[i] = mark  # i is new entry in column j
            Ci[nz] = i  # add i to pattern of C(:,j)
            nz += 1
            if x is not None:
                x[i] = beta * Ax[p]  # x(i) = beta*A(i,j)
        elif x is not None:
            x[i] += beta * Ax[p]  # i exists in C(:,j) already
    return nz


def cs_cumsum(p, c, n):
    """
    p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c

    @param p: size n+1, cumulative sum of c
    @param c: size n, overwritten with p [0..n-1] on output
    @param n: length of c
    @return: sum (c), null on error
    """
    nz = 0
    nz2 = 0.0
    if p is None or c is None: return -1 # check inputs
    for i in range(n):
        p[i] = nz
        nz += c[i]
        nz2 += c[i]              # also in double to avoid CS_INT overflow
        c[i] = p[i]             # also copy p[0..n-1] back into c[0..n-1]
    p[n] = nz
    return int(nz2)               # return sum (c [0..n-1])


def cs_compress(T: TripletsMat):
    """
    C = compressed-column form of a triplet matrix T. The columns of C are
    not sorted, and duplicate entries may be present in C.

    @param T: triplet matrix
    @return: C if successful, null on error
    """
    if not CS_TRIPLET(T):
        return None  # check inputs
    m, n = T.m, T.n
    Ti, Tj, Tx, nz = T.rows, T.column, T.data, T.nz
    C = cs_spalloc(m, n, nz, Tx is not None, False)  # allocate result
    w = ialloc(n)  # get workspace
    Cp = C.indptr
    Ci = C.indices
    Cx = C.data
    for k in range(nz):
        w[Tj[k]] += 1  # column counts
    cs_cumsum(Cp, w, n)  # column pointers
    for k in range(nz):
        p = w[Tj[k]]
        w[Tj[k]] += 1
        Ci[p] = Ti[k]  # A(i,j) is the pth entry in C
        if Cx is not None:
            Cx[p] = Tx[k]
    return C


class cs_ifkeep(object):
    """
    Interface for cs_fkeep.
    """
    def fkeep(self, i, j, aij, other):
        """
        Function used for entries from a sparse matrix.

        @param i: row index
        @param j: column index
        @param aij: value
        @param other: optional parameter
        @return: if false then aij should be dropped
        """
        pass


def _copy(src, dest, length):
    for i in range(length):
        dest[i] = src[i]


def cs_sprealloc(A: CscMat, nzmax):
    """
    Change the max # of entries a sparse matrix can hold.

    @param A: column-compressed matrix
    @param nzmax: new maximum number of entries
    @return: true if successful, false on error
    """
    if A is None:
        return False

    if nzmax <= 0:
        nzmax = A.indptr[A.n] if CS_CSC(A) else A.nz

    Ainew = ialloc(nzmax)
    length = min(nzmax, len(A.indices))
    _copy(A.indices, Ainew, length)
    A.indices = Ainew
    if CS_TRIPLET(A):
        Apnew = ialloc(nzmax)
        length = min(nzmax, len(A.indptr))
        _copy(A.indptr, Apnew, length)
        A.indptr = Apnew
    if A.data is not None:
        Axnew = xalloc(nzmax)
        length = min(nzmax, len(A.data))
        _copy(A.data, Axnew, length)
        A.data = Axnew
    A.nzmax = nzmax
    return True


def cs_fkeep(A: CscMat, fkeep, other):
    """
    Drops entries from a sparse matrix;

    @param A: column-compressed matrix
    @param fkeep: drop aij if fkeep.fkeep(i,j,aij,other) is false
    @param other: optional parameter to fkeep
    @return: nz, new number of entries in A, -1 on error
    """
    nz = 0
    if not CS_CSC(A):
        return -1  # check inputs
    n, Ap, Ai, Ax = A.n, A.indptr, A.indices, A.data
    for j in range(n):
        p = Ap[j]  # get current location of col j
        Ap[j] = nz  # record new location of col j
        while p < Ap[j + 1]:
            if fkeep.fkeep(Ai[p], j, Ax[p] if Ax is not None else 1, other):
                if Ax is not None:
                    Ax[nz] = Ax[p]  # keep A(i,j)
                Ai[nz] = Ai[p]
                nz += 1
            p += 1
    Ap[n] = nz  # finalize A
    cs_sprealloc(A, 0)  # remove extra space from A
    return nz


class _cs_tol(cs_ifkeep):
    """
    Drop small entries from a sparse matrix.
    """
    def fkeep(self, i, j, aij, other):
        return abs(aij) > float(other)


def cs_droptol(A: CscMat, tol):
    """
    Removes entries from a matrix with absolute value <= tol.

    @param A: column-compressed matrix
    @param tol: drop tolerance
    @return: nz, new number of entries in A, -1 on error
    """
    return cs_fkeep(A, _cs_tol(), tol)  # keep all large entries


def cs_add(A: CscMat, B: CscMat, alpha, beta):
    """
    C = alpha*A + beta*B

    @param A: column-compressed matrix
    @param B: column-compressed matrix
    @param alpha: scalar alpha
    @param beta: scalar beta
    @return: C=alpha*A + beta*B, null on error
    """
    nz = 0
    if not CS_CSC(A) or not CS_CSC(B):
        return None  # check inputs
    if A.m != B.m or A.n != B.n:
        return None
    m, anz, n, Bp, Bx = A.m, A.indptr[A.n], B.n, B.indptr, B.data
    bnz = Bp[n]
    w = ialloc(m)  # get workspace
    values = A.data is not None and Bx is not None
    x = xalloc(m) if values else None  # get workspace
    C = cs_spalloc(m, n, anz + bnz, values, False)  # allocate result
    # Cp, Ci, Cx = C.indptr, C.indices, C.data
    for j in range(n):
        C.indptr[j] = nz  # column j of C starts here
        nz = cs_scatter(A, j, alpha, w, x, j + 1, C, nz)  # alpha*A(:,j)
        nz = cs_scatter(B, j, beta, w, x, j + 1, C, nz)  # beta*B(:,j)
        if values:
            for p in range(C.indptr[j], nz):
                C.data[p] = x[C.indices[p]]
    C.indptr[n] = nz  # finalize the last column of C
    return C  # success; free workspace, return C


def cs_multiply(A: CscMat, B: CscMat):
    """
    Sparse matrix multiplication, C = A*B

    @param A: column-compressed matrix
    @param B: column-compressed matrix
    @return: C = A*B, null on error
    """
    nz = 0
    if not CS_CSC(A) or not CS_CSC(B):
        return None # check inputs
    if A.n != B.m:
        return None
    m = A.m
    anz = A.indptr[A.n]
    n, Bp, Bi, Bx = B.n, B.indptr, B.indices, B.data
    bnz = Bp[n]
    w = ialloc(m)  # get workspace
    values = (A.data is not None) and (Bx is not None)
    x = xalloc(m) if values else None  # get workspace
    C = cs_spalloc(m, n, anz + bnz, values, False)  # allocate result
    Cp = C.indptr
    for j in range(n):
        if nz + m > C.nzmax:
            cs_sprealloc(C, 2 * (C.nzmax) + m)
        Ci = C.indices
        Cx = C.data  # C.i and C.x may be reallocated
        Cp[j] = nz  # column j of C starts here
        for p in range(Bp[j], Bp[j + 1]):
            nz = cs_scatter(A, Bi[p], Bx[p] if Bx is not None else 1, w, x, j + 1, C, nz)
        if values:
            for p in range(Cp[j], nz):
                Cx[p] = x[Ci[p]]
    Cp[n] = nz  # finalize the last column of C
    cs_sprealloc(C, 0)  # remove extra space from C
    return C


def cs_transpose(A: CscMat, values: bool=True) -> CscMat:
    """
    Computes the transpose of a sparse matrix, C =A';

    @param A: column-compressed matrix
    @param values: pattern only if false, both pattern and values otherwise
    @return: C=A', null on error
    """
    if not CS_CSC(A):
        return None  # check inputs
    m, n, Ap, Ai, Ax = A.m, A.n, A.indptr, A.indices, A.data
    C = cs_spalloc(n, m, Ap[n], values and (Ax is not None), False)  # allocate result
    w = ialloc(m)  # get workspace
    Cp, Ci, Cx = C.indptr, C.indices, C.data
    for p in range(Ap[n]):
        w[Ai[p]] += 1  # row counts
    cs_cumsum(Cp, w, m)  # row pointers
    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            q = w[Ai[p]]
            w[Ai[p]] += 1
            Ci[q] = j  # place A(i,j) as entry C(j,i)
            if Cx is not None:
                Cx[q] = Ax[p]
    return C


def cs_norm(A: CscMat):
    """
    Computes the 1-norm of a sparse matrix = max (sum (abs (A))), largest
    column sum.

    @param A: column-compressed matrix
    @return: the 1-norm if successful, -1 on error
    """
    norm = 0
    if not CS_CSC(A) or A.data is None:
        return -1  # check inputs
    n, Ap, Ax = A.n, A.indptr, A.data
    for j in range(n):
        s = 0
        for p in range(Ap[j], Ap[j + 1]):
            s += abs(Ax[p])
        norm = max(norm, s)
    return norm


def sub_matrix(A: CscMat, rows, cols) -> CscMat:
    """
    Get SCS arbitrary sub-matrix
    :param A: CSC matrix
    :param rows: row indices to keep
    :param cols: column indices to keep
    :return: CSC sub-matrix
    """
    n_rows = len(rows)
    n_cols = len(cols)

    found = False
    n = 0
    p = 0
    found_idx = 0

    new_val = [0.0 for a in range(A.nz)]
    new_row_ind = [0 for a in range(A.nz)]
    new_col_ptr = [0 for a in range(A.m + 1)]

    new_col_ptr[p] = 0

    for j in cols:

        for k in range(A.indptr[j], A.indptr[j + 1]):
            # search row_ind[k] in rows
            found = False
            found_idx = 0
            while not found and found_idx < n_rows:
                if A.indices[k] == rows[found_idx]:
                    found = True
                found_idx += 1

            # store the values if the row was found in rows
            if found:  # if the row index is in the designated rows...
                new_val[n] = A.data[k]  # store the value
                new_row_ind[n] = found_idx - 1  # store the index where the original index was found inside "rows"
                n += 1
        p += 1
        new_col_ptr[p] = n

    new_col_ptr[p] = n

    B = CscMat()
    B.data = new_val
    B.indices = new_row_ind
    B.indptr = new_col_ptr
    B.nz = n
    B.nzmax = n
    return B


def cs_print(A: CscMat, brief):
    """
    Prints a sparse matrix.

    @param A: sparse matrix (triplet ot column-compressed)
    @param brief: print all of A if false, a few entries otherwise
    @return: true if successful, false on error
    """
    if A is None:
        stdout.write("(null)\n")
        return False
    m, n, Ap, Ai, Ax = A.m, A.n, A.indptr, A.indices, A.data
    nzmax = A.nzmax
    nz = A.nz
    # stdout.write("CSparse.py Version %d.%d.%d, %s.  %s\n" % (CS_VER, CS_SUBVER,
    #         CS_SUBSUB, CS_DATE, CS_COPYRIGHT))
    if nz < 0:
        stdout.write("%d-by-%d, nzmax: %d nnz: %d, 1-norm: %g\n" % (m, n, nzmax, Ap[n], cs_norm(A)))
        for j in range(n):
            stdout.write("    col %d : locations %d to %d\n" % (j, Ap[j], Ap[j + 1] - 1))
            for p in range(Ap[j], Ap[j + 1]):
                stdout.write("      %d : %g\n" % (Ai[p], Ax[p] if Ax is not None else 1))
                if brief and p > 20:
                    stdout.write("  ...\n")
                    return True
    else:
        stdout.write("triplet: %d-by-%d, nzmax: %d nnz: %d\n" % (m, n, nzmax, nz))
        for p in range(nz):
            stdout.write("    %d %d : %g\n" % (Ai[p], Ap[p], Ax[p] if Ax is not None else 1))
            if brief and p > 20:
                stdout.write("  ...\n")
                return True
    return True
