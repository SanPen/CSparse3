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

from collections.abc import Iterable
from CSparse3.float_numba import *
from CSparse3.csc import CscMat


class CooMat:

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
        Cm, Cn, Cp, Ci, Cx = coo_to_csc(m=self.m, n=self.n, Ti=self.rows, Tj=self.column, Tx=self.data, nz=self.nz)
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

