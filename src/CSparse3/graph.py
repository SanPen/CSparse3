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
from numba.typed import List


# @nb.njit("List(List(i8))(i8, i4[:], i4[:])")
@nb.njit
# @nb.njit("List(i8[:])(i8, i4[:], i4[:])")
def find_islands(node_number, indptr, indices):
    """
    Method to get the islands of a graph
    This is the non-recursive version
    :return: islands list where each element is a list of the node indices of the island
    """

    # Mark all the vertices as not visited
    visited = np.zeros(node_number, dtype=nb.boolean)

    # storage structure for the islands (list of lists)
    islands = List.empty_list(List.empty_list(nb.int64))

    # set the island index
    island_idx = 0

    # go though all the vertices...
    for node in range(node_number):

        # if the node has not been visited...
        if not visited[node]:

            # add new island, because the recursive process has already visited all the island connected to v
            # if island_idx >= len(islands):
            islands.append(List.empty_list(nb.int64))

            # ------------------------------------------------------------------------------------------------------
            # DFS: store in the island all the reachable vertices from current vertex "node"
            #
            # declare a stack with the initial node to visit (node)
            stack = List.empty_list(nb.int64)
            stack.append(node)

            while len(stack) > 0:

                # pick the first element of the stack
                v = stack.pop(0)

                # if v has not been visited...
                if not visited[v]:

                    # mark as visited
                    visited[v] = True

                    # add element to the island
                    islands[island_idx].append(v)

                    # Add the neighbours of v to the stack
                    start = indptr[v]
                    end = indptr[v + 1]
                    for i in range(start, end):
                        k = indices[i]  # get the column index in the CSC scheme
                        if not visited[k]:
                            stack.append(k)
            # ------------------------------------------------------------------------------------------------------

            # increase the islands index, because all the other connected vertices have been visited
            island_idx += 1

    # sort the islands to maintain raccord
    # for island in islands:
    #     island.sort()
    return islands
