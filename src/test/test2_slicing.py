from CSparse3.lil import LilMat


def test2():
    eps = 1e-9
    n = 10

    mat = LilMat(n, n)
    mat[:, :] = 0  # this clear the matrix
    mat[2, 2] = 5
    mat[3, 3] = 6
    mat[2, 6:9] = [6, 7, 8]
    mat[3, 6:9] = 1
    mat[1:5, 4] = [1, 2, 3, 4]
    mat[7, [0, 1, 2, 3, 4]] = [3, 3, 4, 4, 4]

    mat[7:9, 7:10] = 8

    print(mat)

    # print(mat[0:5, :])
    # assert 0 <= mat[3, 3] < eps
    # assert mat[2, 2] - 5 < eps

    # C = mat.to_csc()

    # print(C[])

    # print(C)


if __name__ == '__main__':

    test2()
