from scipy.sparse.linalg import spsolve
from . import _superlu
import numpy as np
from scipy.sparse.linalg import spsolve
options = dict(ColPerm=permc_spec)
x, info = _superlu.gssv(N, A.nnz, A.data, A.indices, A.indptr, b, flag, options=options)
if info != 0:
    warn("Matrix is exactly singular", MatrixRankWarning)
    x.fill(np.nan)
if b_is_vector:
    x = x.ravel()