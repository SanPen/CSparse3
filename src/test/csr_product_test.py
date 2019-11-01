import numpy as np
from time import time
from scipy.sparse import csr_matrix, random, diags
from CSparse.multiply import csr_multiply_ff


np.set_printoptions(linewidth=100000)


def test1(check=True):
    np.random.seed(0)
    k = 50
    m, n = k, k

    A = csr_matrix(random(m, n, density=0.02)) + diags(np.ones(m) * m)
    B = csr_matrix(random(m, n, density=0.02)) + diags(np.ones(m) * n)
    C1 = A.dot(B)
    p, r, IC, JC, C, ibot = csr_multiply_ff(p=A.shape[0], q=A.shape[1], IA=A.indptr, JA=A.indices, A=A.data,
                                            Bm=B.shape[0], r=B.shape[1], IB=B.indptr, JB=B.indices, B=B.data)
    C2 = csr_matrix((C, JC, IC), shape=(p, r))

    test = (C2.todense() == C1.todense()).all()

    print('C1')
    print(C1.todense())
    print('C2')
    print(C2.todense())

    diff = np.abs(C1 - C2).max()

    print('pass:', test, diff)

    return test


if __name__ == '__main__':
    test1(check=True)