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
from CSparse.int_functions import *
from CSparse.float_functions import *
from CSparse.conversions import coo_to_csc
from CSparse.csc import CscMat
from CSparse.graph import find_islands


class LilMat:

    def __init__(self, m=0, n=0):
        self.m = m
        self.n = n
        self.nz = 0
        # in power systems, the rows of a sparse matrix always exist (this is a dictionary of dictionaries with values)
        self.data = [{} for i in range(m)]  # -> [row][column] -> value

    def __getitem__(self, key):
        """
        get element
        :param key: any combination of int, Slice and Iterable
        :return: LiL matrix or int instance
        """

        if isinstance(key, tuple):

            if isinstance(key[0], int) and isinstance(key[1], int):
                # (a, b) -> value
                row = self.data[key[0]]
                if key[1] in row.keys():
                    return row[key[1]]
                else:
                    return 0.0

            elif isinstance(key[0], int) and isinstance(key[1], slice):
                # (a, :) -> row a as a LiL matrix
                val = LilMat(1, self.n)
                val.data = self.data[key[0]]
                return val

            elif isinstance(key[0], slice) and isinstance(key[1], int):
                # (:, b) -> column b as a LiL matrix
                val = LilMat(self.m, 1)
                val.data = [{k: v for k, v in d.items() if k == key[1]} for d in self.data]
                return val

            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                # (:, :) -> self
                return self

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):
                # (a, list_b) -> vector of row a and columns given by list_b as a LiL matrix
                val = LilMat(1, len(key[1]))
                val.data = [{k: v for k, v in self.data[key[0]].items() if k in key[1]}]
                return val

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):
                # (list_a, b) -> vector of column b and rows given by list_a as a LiL matrix
                val = LilMat(len(key[0]), 1)
                val.data = [{k: v for k, v in self.data[a].items() if k == key[1]} for a in key[0]]
                return val

            elif isinstance(key[0], slice) and isinstance(key[1], Iterable):
                # (:, list_b) -> Sub-matrix with the columns given by list_b as a LiL matrix
                val = LilMat(self.m, len(key[1]))
                val.data = [{k: v for k, v in d.items() if k in key[1]} for d in self.data]
                return val

            elif isinstance(key[0], Iterable) and isinstance(key[1], slice):
                # (list_a, :) -> Sub-matrix with the rows given by list_a as a LiL matrix
                val = LilMat(len(key[0]), self.n)
                val.data = [self.data[a] for a in key[0]]
                return val

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):
                # (list_a, list_b)  -> non continuous sub-matrix as a LiL matrix
                val = LilMat(len(key[0]), len(key[1]))
                val.data = [{k: v for k, v in self.data[a].items() if k in key[1]} for a in key[0]]
                return val

        else:
            raise Exception('The indices must be a tuple :/')

    def __setitem__(self, key, new_value):
        """
        set element
        :param key: any combination of int, Slice and Iterable
        :param new_value: float, vector or LiL matrix
        """

        if isinstance(key, tuple):

            if isinstance(key[0], int) and isinstance(key[1], int):
                # (a, b) <- value

                row = self.data[key[0]]
                if key[1] not in row.keys():
                    self.nz += 1
                row[key[1]] = new_value

            elif isinstance(key[0], int) and isinstance(key[1], slice):
                # (a, :) <- row a

                pass

            elif isinstance(key[0], slice) and isinstance(key[1], int):
                # (:, b) <- column b

                pass

            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                # (:, :) <- all
                pass

            elif isinstance(key[0], int) and isinstance(key[1], Iterable):
                # (a, list_b) <- vector of row a and columns given by list_b
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], int):
                # (list_a, b) <- vector of column b and rows given by list_a
                pass

            elif isinstance(key[0], slice) and isinstance(key[1], Iterable):
                # (:, list_b) <- Sub-matrix with the columns given by list_b
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], slice):
                # (list_a, :) <- Sub-matrix with the rows given by list_a
                pass

            elif isinstance(key[0], Iterable) and isinstance(key[1], Iterable):
                # (list_a, list_b)  <- non continuous sub-matrix
                pass

        else:
            raise Exception('The indices must be a tuple :/')

    def __len__(self):
        """
        return the number of non zeros (nnz)
        :return: nnz
        """
        return sum([len(row) for row in self.data])

    def __iadd__(self, other: "LilMat"):
        """
        <+=> implementation
        :param other: LilMat instance
        """
        for i, row2 in enumerate(other.data):
            for j, val in row2.items():
                row = self.data[i]
                if j in row.keys():
                    row[j] += val
                else:
                    row[j] = val

    def __isub__(self, other: "LilMat"):
        """
        <-=> implementation
        :param other: LilMat instance
        """
        for i, row2 in enumerate(other.data):
            for j, val in row2.items():
                row = self.data[i]
                if j in row.keys():
                    row[j] -= val
                else:
                    row[j] = -val

    def to_csc(self) -> "CscMat":

        Ti = ialloc(self.nz)
        Tj = ialloc(self.nz)
        Tx = xalloc(self.nz)
        k = 0
        for i, row in enumerate(self.data):
            for j, val in row.items():
                Ti[k] = i
                Tj[k] = j
                Tx[k] = val
                k += 1

        Cm, Cn, Cp, Ci, Cx = coo_to_csc(m=self.m, n=self.n, Ti=Ti, Tj=Tj, Tx=Tx, nz=self.nz)
        mat = CscMat(m=self.m, n=self.n, nz_max=self.nz)
        mat.indices = Ci
        mat.indptr = Cp
        mat.data = Cx
        return mat

