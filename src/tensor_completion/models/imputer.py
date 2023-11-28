import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_completion.utils import unfold, fold

class Imputer:

    def __init__(self, Y_true, Y_sparse, time_lags, alpha, lambd, theta):
        """Initialize the imputer.

        Args:
            Y_true (np.ndarray): The ground truth tensor of shape (n_sensors, period, repeat).
            Y_sparse (np.ndarray): The sparse tensor with missing values of shape (n_sensors, period, repeat).
            time_lags (np.ndarray): The time lags.
            alpha (float): The weight of each dimension.
            lambd (float): The penalty parameter.
            theta (int): The number of singular values to keep.
        """
        self.Y_true = Y_true
        self.Y_sparse = Y_sparse
        self.time_lags = time_lags
        self.alpha = alpha
        self.lambd = lambd
        self.theta = theta

        self.dim = Y_sparse.shape
        self.n_sensors, self.period, self.repeat = self.dim
        self.total_time = self.period * self.repeat
        self.max_lag = np.max(time_lags)
        self.n_lags = time_lags.shape[0]

        self.test_index = np.where((self.Y_sparse == 0) & (self.Y_true != 0))
        self.missing_index = np.where(unfold(self.Y_sparse, 0) == 0)

        self.init_tensors()

        self.time_matrix = np.zeros((self.n_lags, self.total_time - self.max_lag))
        for i in range(len(self.time_lags)):
            self.time_matrix[i, :] = np.arange(self.max_lag - self.time_lags[i], self.total_time - self.time_lags[i])
        self.time_matrix = self.time_matrix.astype(int)

        self.mape_losses = []
        self.rmse_losses = []
        self.tolerances = []
        self.rhos = []

    def init_tensors(self):
        self.T = np.zeros((3, *self.dim))
        self.X = np.zeros((3, *self.dim))
        self.Y_flat = unfold(self.Y_sparse, 0)
        self.Z = self.Y_flat.copy()

        self.Z[self.missing_index] = np.mean(self.Y_flat[self.Y_flat != 0])
        self.Y_test = self.Y_true[self.test_index]

        self.scaling_factor = 0.001
        self.A = np.random.rand(self.n_sensors, self.n_lags) * self.scaling_factor

        self.norm = np.linalg.norm(self.Y_flat, ord='fro')

    def fit(self, max_iterations=100, epsilon=1e-4, rho=1e-5):
        self.rho = rho

        prev_X_hat = self.Y_flat.copy()

        iteration = 0
        progress_bar = tqdm(total=max_iterations)

        while True:

            # Update pass
            self.update_X()

            X_hat_tensor = np.sum([self.alpha[k] * self.X[k] for k in range(3)], axis=0)
            X_hat = unfold(X_hat_tensor, 0)

            Qa = self.compute_Qa(X_hat)
            self.update_Z(Qa)

            self.T += self.rho * (self.X - fold(self.Z, 0, self.dim))

            # Compute loss and check convergence
            tolerance = np.linalg.norm(X_hat - prev_X_hat, ord='fro') / self.norm
            prev_X_hat = X_hat.copy()

            mape_error = MAPE_loss(X_hat_tensor[self.test_index], self.Y_test)
            rmse_error = RMSE_loss(X_hat_tensor[self.test_index], self.Y_test)

            self.mape_losses.append(mape_error)
            self.rmse_losses.append(rmse_error)
            self.tolerances.append(tolerance)
            self.rhos.append(self.rho)

            self.rho = min(1e5, self.rho * 1.05)

            iteration += 1
            progress_bar.update(1)
            progress_bar.set_description(f'{iteration}/{max_iterations} | Tolerance: {tolerance:.6} | MAPE: {mape_error:.2} | RMSE: {rmse_error:.2}')

            if tolerance < epsilon or iteration >= max_iterations: break

        progress_bar.close()

        mape_error = MAPE_loss(X_hat_tensor[self.test_index], self.Y_test)
        rmse_error = RMSE_loss(X_hat_tensor[self.test_index], self.Y_test)

        print("Training complete.")
        print(f'Tolerance: {tolerance:.6} | MAPE: {mape_error} | RMSE: {rmse_error}')

        return X_hat_tensor

    def update_X(self):
        """Update the X tensor.

        For each of the three modes, update the X tensor as follows:
            X_k = fold_k( D( fold_k(Z) - T_k / rho ) )
        where D is the generalized singular thresholding operator with threshold alpha_k / rho and theta singular values.

        i.e

        D(A) = U diag( max(s - 1_theta . alpha / rho, 0) ) V^T where A = U diag(s) V^T is the SVD of A.
        """
        for k in range(3):
            W = fold(self.Z, 0, self.dim) - self.T[k] / self.rho
            U = generalized_singular_threshold(unfold(W, k), self.theta, self.alpha[k] / self.rho)
            self.X[k] = fold(U, k, self.dim)

    def compute_Qa(self, X_hat):
        """Compute the products (Q_m a_m) with a_m solution of the problem
            min_a || Q_m a - Z_m,[h_d+1:] ||_2^2

        with a_m = Q_m^+ Z_m,[h_d+1:] = (Q_m Q_m^T)^-1 Q_m Z_m,[h_d+1:] as a solution.
        """
        N = np.zeros((self.n_sensors, self.total_time - self.max_lag))
        for m in range(self.n_sensors):
            Qm = X_hat[m, self.time_matrix].T
            self.A[m, :] = np.linalg.pinv(Qm) @ self.Z[m, self.max_lag:]
            N[m, :] = Qm @ self.A[m, :]
        return N

    def update_Z(self, Qa):
        """Update the Z tensor.

        Z_m,[:h_d] = 1/3 * sum_k unfold(X_k + T_k / rho)[:h_d]
        Z_m,[h_d+1:] = 1/3 * 1 / (lambda + rho) * sum_k unfold(X_k + T_k / rho)[h_d+1:] + lambda / (lambda + rho) * Q_m a_m

        As our goal is to retrieve the missing values, we only update the values of Z at the missing indices.
        """

        M = unfold(np.mean(self.rho * self.X + self.T, axis=0), 0)

        self.Z[self.missing_index] = np.append(
            M[:, :self.max_lag] / self.rho,
            1 / (self.rho + self.lambd) * (M[:, self.max_lag:] + self.lambd * Qa),
            axis=1
        )[self.missing_index]