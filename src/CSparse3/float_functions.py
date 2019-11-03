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
from CSparse3.int_functions import ialloc, _copy_i


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
    :param An: number of columns
    :param Aindptr: csc column pointers
    :param Aindices: csc row indices
    :param Adata: csc data
    :param nzmax:new maximum number of entries
    :return: indices, data, nzmax
    """

    if nzmax <= 0:
        nzmax = Aindptr[An]

    length = min(nzmax, len(Aindices))
    Ainew = np.empty(nzmax, dtype=nb.int32)
    for i in range(length):
        Ainew[i] = Aindices[i]

    length = min(nzmax, len(Adata))
    Axnew = np.empty(nzmax, dtype=nb.float64)
    for i in range(length):
        Axnew[i] = Adata[i]

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
    :return: new value of nz, -1 on error, x and w are modified
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


@nb.njit("i8(i4[:], i4[:], f8[:], i8, f8, i4[:], f8[:], i8, i4[:], i8)")
def csc_scatter_ff(Aindptr, Aindices, Adata, j, beta, w, x, mark, Ci, nz):
    """
    Scatters and sums a sparse vector A(:,j) into a dense vector, x = x + beta * A(:,j)
    :param Aindptr:
    :param Aindices:
    :param Adata:
    :param j: the column of A to use
    :param beta: scalar multiplied by A(:,j)
    :param w: size m, node i is marked if w[i] = mark
    :param x: size m, ignored if null
    :param mark: mark value of w
    :param Ci: pattern of x accumulated in C.i
    :param nz: pattern of x placed in C starting at C.i[nz]
    :return: new value of nz, -1 on error, x and w are modified
    """

    for p in range(Aindptr[j], Aindptr[j + 1]):
        i = Aindices[p]  # A(i,j) is nonzero
        if w[i] < mark:
            w[i] = mark  # i is new entry in column j
            Ci[nz] = i  # add i to pattern of C(:,j)
            nz += 1
            x[i] = beta * Adata[p]  # x(i) = beta*A(i,j)
        else:
            x[i] += beta * Adata[p]  # i exists in C(:,j) already
    return nz