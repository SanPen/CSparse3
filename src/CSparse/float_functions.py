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
from CSparse.int_functions import ialloc, _copy_i


@nb.njit("f8[:](i8)")
def xalloc(n):
    return np.zeros(n, dtype=nb.float64)


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i8)")
def csc_spalloc_f(m, n, nzmax):
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


@nb.njit("(f8[:], f8[:], i8)")
def _copy_f(src, dest, length):
    for i in range(length):
        dest[i] = src[i]


@nb.njit("Tuple((i4[:], f8[:], i8))(i8, i4[:], i4[:], f8[:], i8)")
def csc_sprealloc_f(An, Aindptr, Aindices, Adata, nzmax):
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


@nb.njit("i8(i4[:], i4[:], f8[:], i8, f8, i4[:], f8[:], i8, i4[:], i8)")
def csc_scatter_f(Ap, Ai, Ax, j, beta, w, x, mark, Ci, nz):
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