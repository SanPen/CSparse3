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

from csparse3 import CscMat
from scipy.sparse import csc_matrix


def scipy_to_mat(scipy_mat: csc_matrix):

    mat = CscMat()

    mat.m, mat.n = scipy_mat.shape
    mat.nz = -1
    mat.data = scipy_mat.data.tolist()
    mat.indices = scipy_mat.indices.tolist()
    mat.indptr = scipy_mat.indptr.tolist()
    mat.nzmax = scipy_mat.nnz

    return mat

