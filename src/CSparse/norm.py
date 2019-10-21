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
