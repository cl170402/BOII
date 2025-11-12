import sobol_seq
import numpy as np

# Generate 5 points in 5-dimensional space (for continuous variables)
sobol_points = sobol_seq.i4_sobol_generate(5, 5)  # 5 variables × 5 points
# Scale to your ranges:
T_vals  = 30 + sobol_points[:, 0] * (40 - 30)
pH_vals = 6  + sobol_points[:, 1] * (8  - 6)
F1_vals = 0  + sobol_points[:, 2] * (50 - 0)
F2_vals = 0  + sobol_points[:, 3] * (50 - 0)
F3_vals = 0  + sobol_points[:, 4] * (50 - 0)

# Randomly assign cell types
cell_types = np.random.choice(['celltype_1','celltype_2','celltype_3'], size=5)

# Combine into one array
X_initial = np.column_stack([T_vals, pH_vals, F1_vals, F2_vals, F3_vals, cell_types]).tolist()

print(X_initial)
