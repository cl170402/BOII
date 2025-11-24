# MLCE_groupname_BO.py
# ------------------------------------------------------------
# MLCE Coursework 2025 – Batch Bayesian Optimisation
# ------------------------------------------------------------
# This script:
#   - Uses Gaussian Process helpers from gaussian_process2.py
#   - Runs batch BO with EI (exploitative settings)
#   - Enumerates all 3 cell types per Sobol point
#   - Respects a 60 s time budget
#   - Prints the best titre and total optimisation time
# ------------------------------------------------------------

# ============ GROUP INFO (EDIT THESE) =======================
group_names     = ['Your Name']
cid_numbers     = ['00000000']
oral_assessment = [1]


# ============ IMPORTS =======================================
import numpy as np
import random
from datetime import datetime
from scipy.stats import norm
import sobol_seq

from MLCE_CWBO2025.gp_model import GP_model

# Import NEW GP helpers (your updated GP file)
from gaussian_process2 import (
    X_lab_to_GP,
    objective_func,
    generate_initial_design_lab
)


# ============================================================
# 1) CANDIDATE GENERATOR FOR BO
# ============================================================
def generate_candidate_batch_lab(n_cand=400):
    """
    Generate candidate points in LAB format.

    Strategy:
    - Draw n_cand Sobol points for the 5 continuous variables (T, pH, F1, F2, F3).
    - For EACH Sobol point, enumerate ALL 3 cell types:
        => total number of candidates = 3 * n_cand

    This is more systematic than assigning random cell types and helps
    the BO explore the categorical space more efficiently.
    """
    # Sobol samples in [0,1]^5
    sobol_points = sobol_seq.i4_sobol_generate(5, n_cand)  # (n_cand, 5)

    X_cand_lab = []

    for i in range(n_cand):
        # Scale to experimental ranges
        T  = 30.0 + sobol_points[i, 0] * 10.0   # T in [30, 40]
        pH = 6.0  + sobol_points[i, 1] * 2.0    # pH in [6, 8]
        F1 =        sobol_points[i, 2] * 50.0   # F1 in [0, 50]
        F2 =        sobol_points[i, 3] * 50.0   # F2 in [0, 50]
        F3 =        sobol_points[i, 4] * 50.0   # F3 in [0, 50]

        # Enumerate ALL cell types for this continuous point
        for cell in ['celltype_1', 'celltype_2', 'celltype_3']:
            X_cand_lab.append([
                float(T),
                float(pH),
                float(F1),
                float(F2),
                float(F3),
                cell
            ])

    # Total candidates: 3 * n_cand
    return X_cand_lab


# ============================================================
# 2) EXPECTED IMPROVEMENT (EI) – MORE EXPLOITATIVE
# ============================================================
def acquisition_ei(X_cand_GP, gp, y_best, xi=0.05):
    """
    Expected Improvement for maximisation.

    Parameters:
        X_cand_GP : (N, d) candidate points (GP numeric space)
        gp        : trained GP_model instance
        y_best    : best observed titre so far
        xi        : exploration parameter (small = more exploitation)

    With a small xi, EI focuses more on high predicted mean while still
    rewarding uncertainty through sigma.
    """
    N = X_cand_GP.shape[0]
    acq = np.zeros(N)

    for i in range(N):
        x_i = X_cand_GP[i]
        mean_vec, var_vec = gp.GP_inference_np(x_i)

        mu   = float(mean_vec[0])
        var  = max(float(var_vec[0]), 0.0)
        sigma = np.sqrt(var)

        if sigma < 1e-12:
            acq[i] = 0.0
            continue

        improvement = mu - y_best - xi
        z = improvement / sigma
        acq[i] = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    return acq


# ============================================================
# 3) BATCH BAYESIAN OPTIMISATION CLASS
# ============================================================
class BO:
    """
    Batch Bayesian Optimisation tuned for:
    - high max titre within 60 seconds,
    - relatively exploitative behaviour (small xi),
    - better categorical exploration (all 3 cell types per Sobol point).

    Uses:
      - Initial Sobol design (from gaussian_process2)
      - RBF GP (GP_model) with multi_hyper restarts
      - EI acquisition (xi small)
      - ε-greedy exploration (small epsilon)
      - light diversity within each batch
    """

    def __init__(self,
                 max_iters=15,       # upper bound; time_budget may stop earlier
                 batch_size=5,
                 n_init=6,
                 n_cand=5000,        # number of Sobol points -> 3 * n_cand candidates
                 multi_hyper=1,
                 seed=None,         # DEBUG ONLY – final submission: seed=None
                 time_budget=60):   # seconds

        # Record start time of the whole optimisation
        self.start_wall = datetime.timestamp(datetime.now())

        # For debugging only; do NOT set seed in final submission
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Logs
        self.X_lab = []   # LAB-format experiments
        self.Y     = []   # observed titres
        self.time  = []   # elapsed time markers

        # ----------------- INITIAL DESIGN --------------------
        start_batch = datetime.timestamp(datetime.now())

        X_init = generate_initial_design_lab(n_init=n_init)
        Y_init = objective_func(X_init).flatten().tolist()

        self.X_lab = list(X_init)
        self.Y     = list(Y_init)

        X_GP_init = X_lab_to_GP(self.X_lab)
        Y_np_init = np.array(self.Y).reshape(-1, 1)

        # GP: multi_hyper = 1 for speed / more aggressive fit
        self.gp = GP_model(
            X_GP_init,
            Y_np_init,
            kernel='RBF',
            multi_hyper=multi_hyper,
            var_out=True
        )

        elapsed = datetime.timestamp(datetime.now()) - start_batch
        self.time += [elapsed] + [0.0] * (n_init - 1)

        # ------------------- BO LOOP -------------------------
        for it in range(max_iters):

            # Stop if we exceed the global time budget
            if (time_budget is not None) and (sum(self.time) > time_budget):
                print(f"Stopping because time_budget = {time_budget}s was reached.")
                break

            start_batch = datetime.timestamp(datetime.now())

            # (a) Propose new batch
            X_batch_lab = self._propose_batch(
                n_cand=n_cand,
                batch_size=batch_size
            )

            # (b) Evaluate batch
            Y_batch = objective_func(X_batch_lab).flatten().tolist()

            # (c) Add data
            self.X_lab += X_batch_lab
            self.Y     += Y_batch

            # (d) Refit GP on all data
            X_GP_all = X_lab_to_GP(self.X_lab)
            Y_np_all = np.array(self.Y).reshape(-1, 1)

            self.gp = GP_model(
                X_GP_all,
                Y_np_all,
                kernel='RBF',
                multi_hyper=multi_hyper,
                var_out=True
            )

            # (e) Update time log
            elapsed = datetime.timestamp(datetime.now()) - start_batch
            self.time += [elapsed] + [0.0] * (batch_size - 1)

        # store total optimisation time
        self.total_time = sum(self.time)


    # --------------------------------------------------------
    # INTERNAL: propose one batch
    # --------------------------------------------------------
    def _propose_batch(self, n_cand, batch_size):
        """
        Propose a batch of new LAB-format points.

        Strategy:
          - 5% of the time: pure random exploration (ε-greedy).
          - 95% of the time: EI with small xi (exploitative).
          - light diversity in normalised GP space to avoid identical points.
        """
        # Generate candidates: 3 * n_cand points (all cell types)
        X_cand_lab = generate_candidate_batch_lab(n_cand=n_cand)
        X_cand_GP  = X_lab_to_GP(X_cand_lab)

        # ε-greedy exploration (small epsilon)
        explore_prob = 0.05
        if np.random.rand() < explore_prob:
            idx = np.random.choice(len(X_cand_lab), size=batch_size, replace=False)
            return [X_cand_lab[i] for i in idx]

        # EI-based selection (small xi for exploitation)
        y_best = max(self.Y)
        acq = acquisition_ei(X_cand_GP, self.gp, y_best, xi=0.01)
        order = np.argsort(-acq)  # best EI first

        # Light diversity using distances in normalised space
        X_min = X_cand_GP.min(axis=0)
        X_max = X_cand_GP.max(axis=0)
        X_range = X_max - X_min + 1e-9
        X_norm = (X_cand_GP - X_min) / X_range

        chosen = []
        min_dist = 0.15  # small distance -> allows cluster around promising region

        for i in order:
            if len(chosen) == 0:
                chosen.append(i)
            else:
                dists = np.linalg.norm(X_norm[i] - X_norm[chosen], axis=1)
                if np.min(dists) > min_dist:
                    chosen.append(i)

            if len(chosen) == batch_size:
                break

        # If diversity rule did not fill batch, top-up with remaining best EI points
        if len(chosen) < batch_size:
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                    if len(chosen) == batch_size:
                        break

        return [X_cand_lab[i] for i in chosen]


# ============================================================
# 4) EXECUTION BLOCK (LOCAL TEST ONLY)
# ============================================================
if __name__ == "__main__":
    # For debugging: use a seed; for final submission: seed=None
    BO_m = BO(
        max_iters=15,
        batch_size=5,
        n_init=6,
        n_cand=400,      # 400 Sobol points -> 1200 candidates (3 cell types)
        multi_hyper=1,
        seed=0,          # DEBUG ONLY – remove / set None in submission
        time_budget= 60
    )

    Y_array = np.array(BO_m.Y)
    best_idx = int(np.argmax(Y_array))

    print("\nBest titre found :", float(Y_array[best_idx]))
    print("Best X (LAB)     :", BO_m.X_lab[best_idx])
    print("Total experiments:", len(BO_m.X_lab))
    print("Approx. total optimisation time (s):", BO_m.total_time)