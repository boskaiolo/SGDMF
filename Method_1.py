from Importer import DatasetImporter
from SparseMatrix import SparseMatrix
from Evaluation import Distortion
import numpy as np

"""
First method:
Just compute the numerical derivative

df = [f(x+h) - f(x)]/h
"""


N_LATENT_FACTORS = 10
NU = 0.1
N_ITERATIONS = 10

h = 0.1

dataset = DatasetImporter()
A = SparseMatrix()
A.addValues(dataset.dataset)
M, N = A.shape()
default_val = np.sqrt(3.)/N_LATENT_FACTORS # why? because it gives exactly an average score of 3 for each cell of A
U = default_val * np.ones((M, N_LATENT_FACTORS))
W = default_val * np.ones((N_LATENT_FACTORS, N))


for count in xrange(N_ITERATIONS):

    # Update of all U(i, j)
    for i in xrange(M):
        for j in xrange(N_LATENT_FACTORS):

            print (i, j)

            # Derivative computation
            f_x = Distortion(A, U, W)
            U[i, j] += h
            f_x_h = Distortion(A, U, W)

            derivative = (f_x_h - f_x)/h

            # SGD update step
            U[i, j] = U[i, j] - h - NU * derivative

    # Update of all W(i, j)
    for i in xrange(N_LATENT_FACTORS):
        for j in xrange(N):

            # Derivative computation
            f_x = Distortion(A, U, W)
            W[i, j] += h
            f_x_h = Distortion(A, U, W)

            derivative = (f_x_h - f_x)/h

            # SGD update step
            W[i, j] = W[i, j] - h - NU * derivative

    # This function will be used in every method
    print "Step", count, "-> Distortion =", Distortion(A, U, W)
