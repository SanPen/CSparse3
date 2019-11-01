from time import time
from scipy.sparse import csc_matrix, random, hstack, vstack

from CSparse.csc import pack_4_by_4, Diag, scipy_to_mat


def test_stack_4():
    """

    :return:
    """
    k = 100
    l = 400
    m = 600

    A = csc_matrix(random(k, l, density=0.2))
    B = csc_matrix(random(k, k, density=0.2))
    C = csc_matrix(random(m, l, density=0.2))
    D = csc_matrix(random(m, k, density=0.2))
    t = time()
    E = hstack((vstack((A, C)), vstack((B, D))))
    print('Scipy\t', time() - t)

    A1 = scipy_to_mat(A)
    B1 = scipy_to_mat(B)
    C1 = scipy_to_mat(C)
    D1 = scipy_to_mat(D)
    t = time()
    E1 = pack_4_by_4(A1, B1, C1, D1)
    print('Csparse3\t', time() - t)
    # print(A1)
    # print(B1)
    # print(C1)
    # print(D1)
    # print(E1)

    stack_test = (E.todense() == E1.todense()).all()

    print('Stacking pass:', stack_test)
    assert stack_test

    return True


if __name__ == '__main__':
    test_stack_4()
