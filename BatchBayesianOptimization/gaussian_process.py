import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime
import pandas as pd

def objective_func(X):
	return (np.array(virtual_lab.conduct_experiment(X)))

X_initial = [[35.0, 7.0, 25.0, 25.0, 25.0, 'celltype_2'], 
             [37.5, 6.5, 37.5, 12.5, 37.5, 'celltype_1'], 
             [32.5, 7.5, 12.5, 37.5, 12.5, 'celltype_3'], 
             [33.75, 6.75, 31.25, 6.25, 43.75, 'celltype_2'], 
             [38.75, 7.75, 6.25, 31.25, 18.75, 'celltype_1']]

X_matrix = np.array(X_initial, dtype=object)
print("X_matrix:\n", X_matrix, "\n")

Y_initial = objective_func(X_initial)
Y_matrix = np.array(Y_initial)
print("Y_matrix:\n", np.round(Y_matrix, 2), "\n")

# ===== 6-D mean vector and 6x6 covariance of inputs =====
# Map the categorical 'cell_type' to numeric codes so we can compute covariance
cell_map = {'celltype_1': 1, 'celltype_2': 2, 'celltype_3': 3}
X_num = np.array([
    [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(cell_map[row[5]])]
    for row in X_initial
], dtype=float)

# Mean vector μ (shape: 6,)
mu = X_num.mean(axis=0)

# Sample covariance Σ (shape: 6x6). ddof=1 gives unbiased estimator (divide by n-1).
Sigma = np.cov(X_num.T, ddof=1)

# Round to 2 decimal places for clean output
mu_rounded = np.round(mu, 2)
Sigma_rounded = np.round(Sigma, 2)

print("Mean vector μ (6D):\n", mu_rounded, "\n")
print("Covariance matrix Σ (6x6):\n", Sigma_rounded, "\n")

