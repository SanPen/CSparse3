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
This is the pure python version where the cython code is outlined
CSparse3.py: a Concise Sparse matrix Python package

@author: Timothy A. Davis
@author: Richard Lincoln
@author: Santiago Peñate Vera
"""

import numpy as np
import numba as nb
import math


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:])",
         parallel=False, nogil=True, fastmath=False, cache=True)  # fastmath=True breaks the code
def csc_multiply_ff(Am, An, Ap, Ai, Ax,
                    Bm, Bn, Bp, Bi, Bx):
    """
    Sparse matrix multiplication, C = A*B where A and B are CSC sparse matrices
    :param Am: number of rows in A
    :param An: number of columns in A
    :param Ap: column pointers of A
    :param Ai: indices of A
    :param Ax: data of A
    :param Bm: number of rows in B
    :param Bn: number of columns in B
    :param Bp: column pointers of B
    :param Bi: indices of B
    :param Bx: data of B
    :return: Cm, Cn, Cp, Ci, Cx, Cnzmax
    """
    assert An == Bm
    nz = 0
    m = Am
    anz = Ap[An]
    bnz = Bp[Bn]
    n = Bn

    t = nb
    w = np.zeros(n, dtype=t.int32)  # ialloc(m)  # get workspace
    x = np.empty(n, dtype=t.float64)  # xalloc(m)  # get workspace

    # allocate result
    Cm = m
    Cn = n
    Cnzmax = int(math.sqrt(m)) * anz + bnz  # the trick here is to allocate just enough memory to avoid reallocating
    Cp = np.empty(n + 1, dtype=t.int32)
    Ci = np.empty(Cnzmax, dtype=t.int32)
    Cx = np.empty(Cnzmax, dtype=t.float64)

    for j in range(n):

        # claim more space
        if nz + m > Cnzmax:
            # Ci, Cx, Cnzmax = csc_sprealloc_f(Cn, Cp, Ci, Cx, 2 * Cnzmax + m)
            print('Re-Allocating')
            Cnzmax = 2 * Cnzmax + m
            if Cnzmax <= 0:
                Cnzmax = Cp[An]

            length = min(Cnzmax, len(Ci))
            Cinew = np.empty(Cnzmax, dtype=nb.int32)
            for i in range(length):
                Cinew[i] = Ci[i]
            Ci = Cinew

            length = min(Cnzmax, len(Cx))
            Cxnew = np.empty(Cnzmax, dtype=nb.float64)
            for i in range(length):
                Cxnew[i] = Cx[i]
            Cx = Cxnew

        # column j of C starts here
        Cp[j] = nz

        # perform the multiplication
        for pb in range(Bp[j], Bp[j + 1]):
            for pa in range(Ap[Bi[pb]], Ap[Bi[pb] + 1]):
                ia = Ai[pa]
                if w[ia] < j + 1:
                    w[ia] = j + 1
                    Ci[nz] = ia
                    nz += 1
                    x[ia] = Bx[pb] * Ax[pa]
                else:
                    x[ia] += Bx[pb] * Ax[pa]

        for pc in range(Cp[j], nz):
            Cx[pc] = x[Ci[pc]]

    Cp[n] = nz  # finalize the last column of C

    # cut the arrays to their nominal size nnz
    # Ci, Cx, Cnzmax = csc_sprealloc_f(Cn, Cp, Ci, Cx, 0)
    Cnzmax = Cp[Cn]
    Cinew = Ci[:Cnzmax]
    Cxnew = Cx[:Cnzmax]

    return Cm, Cn, Cp, Cinew, Cxnew, Cnzmax


@nb.njit("Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:])",
         parallel=False, nogil=True, fastmath=False)
def csr_multiply_ff(p, q, IA, JA, A, Bm, r, IB, JB, B):

    t = np
    xb = np.zeros(r, dtype=t.int32)
    x = np.zeros(r, dtype=t.float64)

    ibot = (len(A) + len(B)) * 500  # nnz
    IC = np.zeros(p + 1, dtype=t.int32)
    JC = np.zeros(ibot, dtype=t.int32)
    C = np.zeros(ibot, dtype=t.float64)

    ip = 0
    for i in range(p):
        IC[i] = ip

        for jp in range(IA[i], IA[i+1]):
            j = JA[jp]

            for kp in range(IB[j], IB[j+1]):

                k = JB[kp]

                if xb[k] != i+1:
                    JC[ip] = k
                    ip += 1
                    xb[k] = i
                    x[k] = A[jp] * B[kp]
                else:
                    x[k] += A[jp] * B[kp]

        # if ip > (ibot - r):  # request extra storage
        for vp in range(IC[i], ip):
            v = JC[vp]
            C[vp] = x[v]
        IC[p] = ip

    return p, r, IC, JC, C, ibot


@nb.njit("f8[:](i8, i8, i4[:], i4[:], f8[:], f8[:])", parallel=False)
def csc_mat_vec_ff(m, n, Ap, Ai, Ax, x):
    """
    Sparse matrix times dense column vector, y = A * x.
    :param m: number of rows
    :param n: number of columns
    :param Ap: pointers
    :param Ai: indices
    :param Ax: data
    :param x: vector x (n)
    :return: vector y (m)
    """

    assert n == x.shape[0]

    y = np.zeros(m, dtype=nb.float64)
    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            y[Ai[p]] += Ax[p] * x[j]
    return y