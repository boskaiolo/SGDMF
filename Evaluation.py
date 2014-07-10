import time
import numpy as np
from SparseMatrix import SparseMatrix


def Distortion(A, U, W):
    """
    Compute the distortion ONLY in the non-zero cells of the target matrix A

    returns ||A - (U * W)||^2 evaluated where A isnonzero (L2 norm)

    """
    A_approx = U.dot(W)
    dst = 0.0
    count = 0.0
    for i in A.csr:
        for j in A.getRow(i):
            val = A_approx[i, j] - A.getVal(i, j)
            dst += val * val
            count += 1.0
    return dst/count


def DistortionOfRow(i, A, U, W):
    """
    ||A(i, :) - (U * W)(i, :)||^2 evaluated where A is nonzero (L2 norm)

    """
    # Get the row i of U
    U_i = U[i]

    # Approximation of the row "i" of A
    A_i_approx = U_i.dot(W)

    # Distortion
    dst = 0.0
    count = 0.0
    for j in A.getRow(i):
        val = A_i_approx[j] - A.getVal(i, j)
        dst += val * val
        count += 1.0
    return dst/count

def DistortionOfColumn(j, A, U, W):
    """
    ||A(:, j) - (U * W)(:, j)||^2 evaluated where A is nonzero (L2 norm)

    """
    # Get the column i of W
    W_j = np.matrix(W).transpose()[j].getA()[0]

    # Approximation of the row "i" of A
    A_j_approx = U.dot(W_j)

    # Distortion
    dst = 0.0
    count = 0.0
    for i in A.getCol(j):
        val = A_j_approx[i] - A.getVal(i, j)
        dst += val * val
        count += 1.0
    return dst/count


class ClockTimer():
    def __init__(self):
        self.clock = time.time()

    def getSpentTime(self):
        return time.time() - self.clock



if __name__ == "__main__":
    ck = ClockTimer()

    matrix = SparseMatrix()
    matrix.addValue(0, 0, 1.0)
    matrix.addValue(1, 1, 2.0)
    matrix.addValue(2, 1, 2.0)
    matrix.addValue(3, 2, 2.0)
    matrix.addValue(4, 0, 2.0)
    matrix.addValue(4, 2, 12.0)

    U = np.ones((5, 2))
    W = np.ones((2, 3))

    d = Distortion(matrix, U, W)
    print (d)

    print "I took", ck.getSpentTime(), "seconds"
