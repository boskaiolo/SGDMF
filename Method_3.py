import random
from Importer import DatasetImporter
from Evaluation import *

"""
Just a randomized version of the second
"""

N_LATENT_FACTORS = 10
NU = 0.1
N_ITERATIONS = 10

h = 0.1

dataset = DatasetImporter()
A = SparseMatrix()
A.addValues(dataset.dataset)
M, N = A.shape()

default_val = np.sqrt(3.) / N_LATENT_FACTORS  # why? because it gives exactly an average score of 3 for each cell of A
U = default_val * np.ones((M, N_LATENT_FACTORS))
W = default_val * np.ones((N_LATENT_FACTORS, N))

for count in xrange(N_ITERATIONS * (M * N_LATENT_FACTORS + N_LATENT_FACTORS * N)):

    val = random.randint(0, M * N_LATENT_FACTORS + N_LATENT_FACTORS * N - 1)

    if val < M * N_LATENT_FACTORS:
        i = random.randint(0, M - 1)
        j = random.randint(0, N_LATENT_FACTORS - 1)

        # Derivative computation
        f_x = DistortionOfRow(i, A, U, W)
        U[i, j] += h
        f_x_h = DistortionOfRow(i, A, U, W)

        derivative = (f_x_h - f_x) / h

        # SGD update step
        U[i, j] = U[i, j] - h - NU * derivative

    else:
        i = random.randint(0, N_LATENT_FACTORS - 1)
        j = random.randint(0, N - 1)

        # Derivative computation
        f_x = DistortionOfColumn(j, A, U, W)
        W[i, j] += h
        f_x_h = DistortionOfColumn(j, A, U, W)

        derivative = (f_x_h - f_x) / h

        # SGD update step
        W[i, j] = W[i, j] - h - NU * derivative

    if count % (M * N_LATENT_FACTORS + N_LATENT_FACTORS * N) == (M * N_LATENT_FACTORS + N_LATENT_FACTORS * N) - 1:
        # This function will be used in every method
        print "Step", count / ((M * N_LATENT_FACTORS + N_LATENT_FACTORS * N) + 1), "-> Distortion =", Distortion(A, U,
                                                                                                                 W)
