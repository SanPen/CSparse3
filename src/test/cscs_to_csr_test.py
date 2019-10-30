import numpy as np
from CSparse.csc import CscMat


def test_csc_to_csr():
    """
    Test the conversion from CSC to CSR format
    :return: True if successful
    """

    # declare the CSC data
    A = CscMat(6, 3)
    A.data = np.array([4, 3, 3, 9, 7, 8, 4, 8, 8, 9], dtype=np.float)
    A.indices = np.array([0, 1, 3, 1, 2, 4, 5, 2, 3, 4], dtype=np.int32)
    A.indptr = np.array([0, 3, 7, 10], dtype=np.int32)

    print(A)

    # convert
    Bp, Bi, Bx = A.to_csr()

    # declare the check data
    csr_data = np.array([4, 3, 9, 7, 8, 3, 8, 8, 9, 4], dtype=np.float)
    csr_indices = np.array([0, 0, 1, 1, 2, 0, 2, 1, 2, 1], dtype=np.int32)
    csr_indptr = np.array([0, 1, 3, 5, 7, 9, 10], dtype=np.int32)

    # check the conversion
    assert (Bp == csr_indptr).all()
    assert (Bi == csr_indices).all()
    assert (Bx == csr_data).all()

    return True


if __name__ == '__main__':
    test_csc_to_csr()
