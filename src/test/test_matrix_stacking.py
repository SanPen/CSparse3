from scipy.sparse import csc_matrix, random

from CSparse.csc import pack_4_by_4, Diag, scipy_to_mat

n = 5
# A = Diag(5, 5, 1.0)
# B = Diag(5, 5, 2.0)
# C = Diag(5, 5, 3.0)
# D = Diag(5, 5, 4.0)

A = csc_matrix(random(5, 4, density=0.2))
B = csc_matrix(random(5, 5, density=0.2))
C = csc_matrix(random(6, 4, density=0.2))
D = csc_matrix(random(6, 5, density=0.2))

A1 = scipy_to_mat(A)
B1 = scipy_to_mat(B)
C1 = scipy_to_mat(C)
D1 = scipy_to_mat(D)

print(A1)
print(B1)
print(C1)
print(D1)

E = pack_4_by_4(A1, B1, C1, D1)

print(E)