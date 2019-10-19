import pyximport
pyximport.install()

import numpy as np
from time import time
from scipy.sparse import csc_matrix, random

from cython.CSparse3 import scipy_to_mat

np.set_printoptions( linewidth=100000)


def test1():
    np.random.seed(0)
    m, n = 10, 10

    A = csc_matrix(random(m, n, density=0.2))
    B = csc_matrix(random(m, n, density=0.2))
    x = np.random.random(m)

    # ---------------------------------------------------------------------
    # Scipy
    # ---------------------------------------------------------------------
    t = time()
    C = A + B
    D = A - B
    F = A * B
    G = C * x
    H = A * 5
    print('Scipy\t', time() - t)

    # ---------------------------------------------------------------------
    # CSparse3
    # ---------------------------------------------------------------------
    t = time()
    A2 = scipy_to_mat(A)
    B2 = scipy_to_mat(B)
    C2 = A2 + B2
    D2 = A2 - B2
    F2 = A2 * B2
    G2 = C2 * x
    H2 = A2 * 5.0
    print('CSparse\t', time() - t)
    # ---------------------------------------------------------------------
    # check
    # ---------------------------------------------------------------------

    pass_sum = (C.todense() == C2.todense()).all()
    pass_subt = (D.todense() == D2.todense()).all()
    pass_mult = (F.todense() == F2.todense()).all()
    pass_mat_vec = (G == G2).all()
    pass_mult_scalar = (H.todense() == H2.todense()).all()

    assert pass_sum
    assert pass_subt
    assert pass_mult
    assert pass_mat_vec
    assert pass_mult_scalar

    print('+\t\t', pass_sum)
    print('-\t\t', pass_subt)
    print('mat mat\t', pass_mult)
    print('mat vec\t', pass_mat_vec)
    print('scalar *', pass_mult_scalar)
    pass


if __name__ == '__main__':
    test1()