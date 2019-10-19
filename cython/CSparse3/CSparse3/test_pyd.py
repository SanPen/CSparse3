from csparse3 import CscMat
from utils import scipy_to_mat
from scipy.sparse import csc_matrix, random
import numpy as np
np.set_printoptions(linewidth=100000)


def test1():

    m, n = 10, 10

    A = csc_matrix(random(m, n, density=0.2))
    B = csc_matrix(random(m, n, density=0.2))

    # Scipy
    C = A + B
    D = A - B
    E = A * B
    F = A.dot(B)

    # same operations with CSparse3
    A2 = scipy_to_mat(A)
    B2 = scipy_to_mat(B)

    C2 = A2 + B2
    D2 = A2 - B2
    F2 = A2 * B2

    print(F.todense())
    print()
    print(F2)

    pass


if __name__ == '__main__':
    test1()