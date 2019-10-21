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
from CSparse.float_functions import csc_spalloc_f


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:], i8)")
def coo_to_csc(m, n, Ti, Tj, Tx, nz):
    """
    C = compressed-column form of a triplet matrix T. The columns of C are
    not sorted, and duplicate entries may be present in C.

    @param T: triplet matrix
    @return: Cm, Cn, Cp, Ci, Cx
    """

    Cm, Cn, Cp, Ci, Cx, nz = csc_spalloc_f(m, n, nz)  # allocate result

    w = ialloc(n)  # get workspace

    for k in range(nz):
        w[Tj[k]] += 1  # column counts

    csc_cumsum_i(Cp, w, n)  # column pointers

    for k in range(nz):
        p = w[Tj[k]]
        w[Tj[k]] += 1
        Ci[p] = Ti[k]  # A(i,j) is the pth entry in C
        # if Cx is not None:
        Cx[p] = Tx[k]

    return Cm, Cn, Cp, Ci, Cx


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

