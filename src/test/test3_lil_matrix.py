import numpy as np
from CSparse3 import LilMat

'''
name	    bus_from	bus_to	active	rate	mttf	mttr	R	    X	    G	    B
Branch 1	Bus 3	    Bus 1	TRUE	70	    0	    0	    0.05	0.11	1E-20	0.02
Branch 1	Bus 4	    Bus 3	TRUE	18	    0	    0	    0.06	0.13	1E-20	0.03
Branch 1	Bus 5	    Bus 4	TRUE	20	    0	    0	    0.04	0.09	1E-20	0.02
Branch 1	Bus 5	    Bus 2	TRUE	10	    0	    0	    0.04	0.09	1E-20	0.02
Branch 1	Bus 5	    Bus 1	TRUE	90	    0	    0	    0.03	0.08	1E-20	0.02
Branch 1	Bus 2	    Bus 1	TRUE	60	    0	    0	    0.05	0.11	1E-20	0.02
Branch 1	Bus 2	    Bus 3	TRUE	20	    0	    0	    0.04	0.09	1E-20	0.02
'''


# line data format:
#             From | To | R (p.u.) | X (p.u.) | B (p.u.)
line_data = [[3,     1,   0.05,   0.11,   0.02],
             [4,     3,   0.06,   0.13,   0.03],
             [5,     4,   0.04,   0.09,   0.02],
             [5,     2,   0.04,   0.09,   0.02],
             [5,     1,   0.03,   0.08,   0.02],
             [2,     1,   0.05,   0.11,   0.02],
             [2,     3,   0.04,   0.09,   0.02]]
line_data = np.array(line_data, dtype=np.object)

n = 5
m = len(line_data)
f_mat = LilMat(m, n)
t_mat = LilMat(m, n)

k = 0
for f, t, r, x, b in line_data:

    F = f - 1
    T = t - 1
    f_mat[k, F] = 1
    t_mat[k, T] = 1
    k += 1

C = f_mat.to_csc() - t_mat.to_csc()
print(C)

A = C * C.t()
print(A)

islands = A.islands()
for i in islands:
    print(i)
