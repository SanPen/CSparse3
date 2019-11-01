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
import numpy as np
import numba as nb


@nb.njit("Tuple((i4[:], i4[:], f8[:]))(i8, f8)")
def csc_diagonal(m, value=1.0):
    """

    :param m:
    :param value:
    :return:
    """
    indptr = np.empty(m + 1, dtype=np.int32)
    indices = np.empty(m, dtype=np.int32)
    data = np.empty(m, dtype=np.float64)
    for i in range(m):
        indptr[i] = i
        indices[i] = i
        data[i] = value
    indptr[m] = m

    return indices, indptr, data


@nb.njit("Tuple((i4[:], i4[:], f8[:]))(i8, f8[:])")
def csc_diagonal_from_array(m, array):
    """

    :param m:
    :param array:
    :return:
    """
    indptr = np.empty(m + 1, dtype=np.int32)
    indices = np.empty(m, dtype=np.int32)
    data = np.empty(m, dtype=np.float64)
    for i in range(m):
        indptr[i] = i
        indices[i] = i
        data[i] = array[i]
    indptr[m] = m

    return indices, indptr, data


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:]))"
         "(i8, i8, i4[:], i4[:], f8[:], "
         "i8, i8, i4[:], i4[:], f8[:], "
         "i8, i8, i4[:], i4[:], f8[:], "
         "i8, i8, i4[:], i4[:], f8[:])",
         parallel=False, nogil=True, fastmath=True)
def stack_4_by_4_ff(am, an, Ai, Ap, Ax,
                    bm, bn, Bi, Bp, Bx,
                    cm, cn, Ci, Cp, Cx,
                    dm, dn, Di, Dp, Dx):
    """
    stack csc sparse float matrices like this:
    | A | B |
    | C | D |

    :param am:
    :param an:
    :param Ai:
    :param Ap:
    :param Ax:
    :param bm:
    :param bn:
    :param Bi:
    :param Bp:
    :param Bx:
    :param cm:
    :param cn:
    :param Ci:
    :param Cp:
    :param Cx:
    :param dm:
    :param dn:
    :param Di:
    :param Dp:
    :param Dx:
    :return:
    """

    # check dimensional compatibility
    assert am == bm
    assert cm == dm
    assert an == cn
    assert bn == dn

    nnz = Ap[an] + Bp[bn] + Cp[cn] + Dp[dn]

    m = am + cm
    n = an + bn

    indptr = np.zeros(n + 1, dtype=np.int32)
    indices = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float64)
    cnt = 0
    indptr[0] = 0
    for j in range(an):  # for every column, same as range(cols + 1) For A and C
        for k in range(Ap[j], Ap[j + 1]):  # for every entry in the column from A
            indices[cnt] = Ai[k]  # row index
            data[cnt] = Ax[k]
            cnt += 1

        for k in range(Cp[j], Cp[j + 1]):  # for every entry in the column from C
            indices[cnt] = Ci[k] + am  # row index
            data[cnt] = Cx[k]
            cnt += 1

        indptr[j + 1] = cnt

    for j in range(bn):  # for every column, same as range(cols + 1) For B and D
        for k in range(Bp[j], Bp[j + 1]):  # for every entry in the column from B
            indices[cnt] = Bi[k]  # row index
            data[cnt] = Bx[k]
            cnt += 1

        for k in range(Dp[j], Dp[j + 1]):  # for every entry in the column from D
            indices[cnt] = Di[k] + bm  # row index
            data[cnt] = Dx[k]
            cnt += 1

        indptr[an + j + 1] = cnt

    return m, n, indices, indptr, data


