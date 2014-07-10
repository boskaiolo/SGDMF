
class SparseMatrix:
    """
    Sparse matrix with the capability of returning filled cells in rows and columns

    Example:

        [1    ]
        [  2  ]
    M = [  3  ]
        [    4]
        [5   6]

    [Construct the matrix M]
    M.getRow(4) = {0: 5, 2: 6} (that's the sparse representation of the first row
    M.getCol(1) = {1: 2, 2: 3} (that's the sparse representation of the second column)

    """

    def __init__(self):
        self.csr = {}
        self.csc = {}

    def addValue(self, i, j, val):
        assert(type(i) == int and type(j) == int)
        try:
            self.csr[i][j] = val
        except KeyError:
            self.csr[i] = {}
            self.csr[i][j] = val

        try:
            self.csc[j][i] = val
        except KeyError:
            self.csc[j] = {}
            self.csc[j][i] = val

    def addValues(self, list_of_elements):
        assert (type(list_of_elements) == list)
        for element in list_of_elements:
            assert(type(element) == tuple)
            self.addValue(element[0], element[1], element[2])

    def getRow(self, row):
        return self.csr.get(row, {})

    def getCol(self, col):
        return self.csc.get(col, {})

    def getVal(self, row, col):
        # This method might raise KeyError exception
        return self.csr[row][col]

    def shape(self):
        return max(self.csr.keys())+1, max(self.csc.keys())+1


if __name__ == "__main__":
    matrix = SparseMatrix()
    matrix.addValue(0, 0, 1.0)
    matrix.addValue(1, 1, 2.0)
    matrix.addValue(2, 1, 3.0)
    matrix.addValue(3, 2, 4.0)
    matrix.addValue(4, 0, 5.0)
    matrix.addValue(4, 2, 6.0)
    print(matrix.getRow(4))
    print(matrix.getCol(1))
    print(matrix.shape())

    matrix.addValues([(10, 10, 10.0)])
    print(matrix.getRow(6))
    print(matrix.getCol(6))
    print(matrix.shape())