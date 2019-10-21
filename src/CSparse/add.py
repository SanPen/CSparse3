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

import numba as nb
from CSparse.int_functions import ialloc
from CSparse.float_functions import xalloc, csc_spalloc_f, csc_scatter_f


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:], f8, f8)")
def csc_add_ff(Am, An, Aindptr, Aindices, Adata,
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

    m, anz, n, Bp, Bx = Am, Aindptr[An], Bn, Bindptr, Bdata

    bnz = Bp[n]

    w = ialloc(m)  # get workspace

    x = xalloc(m)   # get workspace

    Cm, Cn, Cp, Ci, Cx, Cnzmax = csc_spalloc_f(m, n, anz + bnz)  # allocate result

    for j in range(n):
        Cp[j] = nz  # column j of C starts here

        nz = csc_scatter_f(Aindptr, Aindices, Adata, j, alpha, w, x, j + 1, Ci, nz)  # alpha*A(:,j)

        nz = csc_scatter_f(Bindptr, Bindices, Bdata, j, beta, w, x, j + 1, Ci, nz)  # beta*B(:,j)

        for p in range(Cp[j], nz):
            Cx[p] = x[Ci[p]]

    Cp[n] = nz  # finalize the last column of C

    return Cm, Cn, Cp, Ci, Cx  # success; free workspace, return C