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
        # in power systems, the rows of a sparse matrix always exist
        self.data = [{} for i in range(m)]

    def __getitem__(self, coord):
        """
        get element
        :param coord:
        :return:
        """
        row = self.data[coord[0]]
        if coord[1] in row.keys():
            return row[coord[1]]
        else:
            return 0.0

    def __setitem__(self, coord, new_value):
        """
        set element
        :param coord:
        :param new_value:
        :return:
        """
        row = self.data[coord[0]]
        if coord[1] not in row.keys():
            self.nz += 1
        row[coord[1]] = new_value

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

