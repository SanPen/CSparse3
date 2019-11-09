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
from numba.pycc import CC
from numba.typed import List
import math

cc = CC(extension_name='csr_native', source_module='CSparse3')
cc.output_dir = 'CSParse3'


def compile_code():
    cc.compile()


@cc.export("csr_multiply_ff", "Tuple((i8, i8, i4[:], i4[:], f8[:], i8))(i8, i8, i4[:], i4[:], f8[:], i8, i8, i4[:], i4[:], f8[:])")
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

