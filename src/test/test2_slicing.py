from CSparse.lil import LilMat


def test2():
    eps = 1e-9
    n = 10

    T = LilMat(n, n)
    T[:, :] = 0
    T[2, 2] = 5
    T[4, 4] = 6
    T[2, 6:9] = [6, 7, 8]
    T[3, 6:9] = 1
    T[0:4, 4] = [1, 2, 3, 4]
    T[7, [0, 1, 2, 3, 4]] = [3, 3, 4, 4, 4]

    print(T[0:5, :])
    assert 0 <= T[3, 3] < eps
    assert T[2, 2] - 5 < eps



    C = T.to_csc()

    # print(C[])

    print(C)


if __name__ == '__main__':

    test2()
