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
import numba as nb
from CSparse.float_functions import csc_spalloc_f, csc_scatter_f, csc_sprealloc_f


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:])",
         parallel=True)
def csc_multiply_ff(Am, An, Aindptr, Aindices, Adata,
                    Bm, Bn, Bindptr, Bindices, Bdata):
    """
    Sparse matrix multiplication, C = A*B
    :param Am: number of rows in A
    :param An: number of columns in A
    :param Aindptr: column pointers of A
    :param Aindices: indices of A
    :param Adata: data of A
    :param Bm: number of rows in B
    :param Bn: number of columns in B
    :param Bindptr: column pointers of B
    :param Bindices: indices of B
    :param Bdata: data of B
    :return: Cm, Cn, Cp, Ci, Cx, Cnzmax
    """

    nz = 0

    m = Am

    anz = Aindptr[An]

    n, Bp, Bi, Bx = Bn, Bindptr, Bindices, Bdata

    bnz = Bp[n]

    w = np.zeros(n, dtype=nb.int32)  # ialloc(m)  # get workspace

    x = np.zeros(n, dtype=nb.float64)  # xalloc(m)  # get workspace

    Cm, Cn, Cp, Ci, Cx, Cnzmax = csc_spalloc_f(m, n, anz + bnz)  # allocate result

    for j in range(n):

        if nz + m > Cnzmax:
            Ci, Cx, Cnzmax = csc_sprealloc_f(Cn, Cp, Ci, Cx, 2 * Cnzmax + m)

        Cp[j] = nz  # column j of C starts here

        for p in range(Bp[j], Bp[j + 1]):
            nz = csc_scatter_f(Aindptr, Aindices, Adata, Bi[p], Bx[p], w, x, j + 1, Ci, nz)

        for p in range(Cp[j], nz):
            Cx[p] = x[Ci[p]]

    Cp[n] = nz  # finalize the last column of C

    Ci, Cx, Cnzmax = csc_sprealloc_f(Cn, Cp, Ci, Cx, 0)  # remove extra space from C

    return Cm, Cn, Cp, Ci, Cx, Cnzmax


@nb.njit("f8[:](i8, i8, i4[:], i4[:], f8[:], f8[:])", parallel=True)
def csc_mat_vec_ff(m, n, Ap, Ai, Ax, x):
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