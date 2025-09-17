# Modified from https://github.com/kevinsbello/dagma/blob/main/src/dagma/nonlinear.py
# Modifications Copyright (C) 2023 Dan Waxman

import copy
import torch
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import copy
from tqdm.auto import tqdm
from locally_connected import LocallyConnected
import abc
import typing
import math


class Dagma_DCE_Module(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def h_func(self, W1: torch.tensor, W2: torch.tensor, s: float) -> torch.Tensor:  
        ...

    @abc.abstractmethod
    def get_l1_reg(self, W: torch.Tensor) -> torch.Tensor:
        ...


class DagmaDCE:
    def __init__(self, model: Dagma_DCE_Module, use_mle_loss=True):
        """Initializes a DAGMA DCE model. Requires a `DAGMA_DCE_Module`

        Args:
            model (Dagma_DCE_Module): module implementing adjacency matrix,
                h_func constraint, and L1 regularization
            use_mse_loss (bool, optional): to use MSE loss instead of log MSE loss.
                Defaults to True.
        """
        self.model = model
        self.loss = self.mle_loss if use_mle_loss else self.mse_loss

    def mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        """Computes the MSE loss sum (output - target)^2 / (2N)"""
        n, d = target.shape
        return 0.5 / n * torch.sum((output - target) ** 2)

    def mle_loss(self, output: torch.Tensor, target: torch.Tensor, Sigma: torch.Tensor):
        """Computes the MLE loss 1/n*Tr((X-X_est)Sigma^{-1}(X-X_est)^T)"""
        n, d = target.shape
        tmp = torch.linalg.solve(Sigma, (target - output).T)
        mle = torch.trace((target - output)@tmp)/n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle

    def minimize(
        self,
        max_iter: int,
        lr: float,
        lambda1: float,
        lambda2: float,
        mu: float,
        s: float,
        pbar: tqdm,
        lr_decay: bool = False,
        checkpoint: int = 1000,
        tol: float = 1e-3,
    ):
        """Perform minimization using the barrier method optimization

        Args:
            max_iter (int): maximum number of iterations to optimize
            lr (float): learning rate for adam
            lambda1 (float): regularization parameter
            lambda2 (float): weight decay
            mu (float): regularization parameter for barrier method
            s (float): DAMGA constraint hyperparameter
            pbar (tqdm): progress bar to use
            lr_decay (bool, optional): whether or not to use learning rate decay.
                Defaults to False.
            checkpoint (int, optional): how often to checkpoint. Defaults to 1000.
            tol (float, optional): tolerance to terminate learning. Defaults to 1e-3.
        """
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.99, 0.999),
            weight_decay=mu * lambda2,
        )

        obj_prev = 1e16

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8 if lr_decay else 1.0
        )

        for i in range(max_iter):
            optimizer.zero_grad()

            if i == 0:
                X_hat = self.model(self.X)
                Sigma = self.model.get_Sigma()
                score = self.loss(X_hat, self.X, Sigma)
                obj = score

            else:
                W_current, observed_derivs = self.model.get_graph(self.X)
                Sigma = self.model.get_Sigma()
                ##### new test
                # Sigma = torch.eye(self.X.shape[1])
                Wii = torch.diag(torch.diag(Sigma))
                W2 = Sigma - Wii
                h_val = self.model.h_func(W_current, W2, s)

                if h_val.item() < 0:
                    return False

                X_hat = self.model(self.X)
                score = self.mle_loss(X_hat, self.X, Sigma)

                l1_reg = lambda1 * self.model.get_l1_reg(observed_derivs)

                obj = mu * (score + l1_reg) + h_val

                if i % 100 == 0:
                    print("Sigma: ", Sigma)
                    print("obj: ", obj)
                    print("mle loss: ", score)
                    print("h_val: ", h_val)

            obj.backward()
            optimizer.step()

            if lr_decay and (i + 1) % 1000 == 0:
                scheduler.step()

            if i % checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()

                if np.abs((obj_prev - obj_new) / (obj_prev)) <= tol:
                    pbar.update(max_iter - i)
                    break
                obj_prev = obj_new

            pbar.update(1)

        return True

    def fit(
        self,
        X: torch.Tensor,
        lambda1: float = 0.02,
        lambda2: float = 0.005,
        T: int = 4,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        s: float = 1.0,
        warm_iter: int = 5e3,
        max_iter: int = 8e3,
        lr: float = 1e-3,
        disable_pbar: bool = False,
    ) -> torch.Tensor:
        """Fits the DAGMA-DCE model

        Args:
            X (torch.Tensor): inputs
            lambda1 (float, optional): regularization parameter. Defaults to 0.02.
            lambda2 (float, optional): weight decay. Defaults to 0.005.
            T (int, optional): number of barrier loops. Defaults to 4.
            mu_init (float, optional): barrier path coefficient. Defaults to 1.0.
            mu_factor (float, optional): decay parameter for mu. Defaults to 0.1.
            s (float, optional): DAGMA constraint hyperparameter. Defaults to 1.0.
            warm_iter (int, optional): number of warmup models. Defaults to 5e3.
            max_iter (int, optional): maximum number of iterations for learning. Defaults to 8e3.
            lr (float, optional): learning rate. Defaults to 1e-3.
            disable_pbar (bool, optional): whether or not to use the progress bar. Defaults to False.

        Returns:
            torch.Tensor: graph returned by the model
        """
        mu = mu_init
        self.X = X

        with tqdm(total=(T - 1) * warm_iter + max_iter, disable=disable_pbar) as pbar:
            for i in range(int(T)):
                success, s_cur = False, s
                lr_decay = False

                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)

                while success is False:
                    success = self.minimize(
                        inner_iter,
                        lr,
                        lambda1,
                        lambda2,
                        mu,
                        s_cur,
                        lr_decay=lr_decay,
                        pbar=pbar,
                    )

                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break  # lr is too small

                    mu *= mu_factor

            Sigma = self.model.get_Sigma()
            Wii = torch.diag(torch.diag(Sigma))
            W2 = Sigma - Wii
            x_est = self.model(self.X)

        return self.model.get_graph(self.X)[0], W2, x_est
    
    
def SPDLogCholesky(M: torch.tensor)-> torch.Tensor:
    """
    Use LogCholesky decomposition that map a matrix M to a SPD Sigma matrix.
    """
    # Take strictly lower triangular matrix
    M_strict = M.tril(diagonal=-1)
    # Make matrix with exponentiated diagonal
    D = M.diag()
    # Make the Cholesky decomposition matrix
    L = M_strict + torch.diag(torch.exp(D))
    # Invert the Cholesky decomposition
    Sigma = torch.matmul(L, L.t()) #+ 1e-6 * torch.eye(d) # numerical stability
    return Sigma

def reverse_SPDLogCholesky(Sigma: torch.tensor)-> torch.Tensor:
    """
    Reverse the LogCholesky decomposition that map the SPD Sigma matrix to the matrix M.
    """
    # Compute the Cholesky decomposition
    L = torch.linalg.cholesky(Sigma)
    # Take strictly lower triangular matrix
    M_strict = L.tril(diagonal=-1)
    # Take the logarithm of the diagonal
    D = torch.diag(torch.log(L.diag()))
    # Return the log-Cholesky parametrization
    M = M_strict + D
    return M


class DagmaMLP_DCE(Dagma_DCE_Module):
    def __init__(
        self,
        dims: typing.List[int],
        bias: bool = True,
        dtype: torch.dtype = torch.double,
    ):
        """Initializes the DAGMA DCE MLP module

        Args:
            dims (typing.List[int]): dims
            bias (bool, optional): whether or not to use bias. Defaults to True.
            dtype (torch.dtype, optional): dtype to use. Defaults to torch.double.
        """
        torch.set_default_dtype(dtype)

        super(DagmaMLP_DCE, self).__init__()

        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)

        Sigma = torch.eye(self.d, dtype=dtype)
        self.M = reverse_SPDLogCholesky(Sigma)
        self.M = nn.Parameter(self.M)

        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)

        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
            # Each dimension d has a separate weight to learn

        self.fc2 = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sigmoidal feedforward NN

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        x = self.fc1(x) # [n, self.d * dims[1]]

        x = x.view(-1, self.dims[0], self.dims[1]) # [n, d, self.dims[1]]

        self.activation = nn.LeakyReLU()

        for fc in self.fc2:
            # x = torch.sigmoid(x)
            x = self.activation(x)
            x = fc(x) # [n, d, self.dims[2]]

        x = x.squeeze(dim=2) #[n, d]

        return x
    
    def get_Sigma(self)-> torch.Tensor:

        Sigma = SPDLogCholesky(self.M)
        return Sigma

    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Get the adjacency matrix defined by the DCE and the batched Jacobian

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor, torch.Tensor: the weighted graph and batched Jacobian
        """
        x_dummy = x.detach().requires_grad_()

        observed_deriv = torch.func.vmap(torch.func.jacrev(self.forward))(x_dummy).view(
            -1, self.d, self.d
        )#[n, d, d], observed_deriv[i, j, k]=for ith sample, derivative of f_j wrt x_k

        # Adjacency matrix is RMS Jacobian
        W = torch.sqrt(torch.mean(observed_deriv**2, axis=0).T) #[d, d]

        return W, observed_deriv
    
    def get_Hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Get the adjacency matrix defined by the DCE and the batched Jacobian

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor, torch.Tensor: the weighted graph and batched Jacobian
        """
        x_dummy = x.detach().requires_grad_()

        observed_deriv = torch.func.vmap(torch.func.jacrev(self.forward))(x_dummy).view(
            -1, self.d, self.d
        )#[n, d, d], observed_deriv[i, j, k]=for ith sample, derivative of f_j wrt x_k

        z = (torch.randint_like(observed_deriv, 0, 2).float() * 2 - 1) # [n, d, d]

        def g_single(xi, zi):
            J = torch.func.jacrev(self.forward)(xi)   # [d,d]
            return (J * zi).sum(dim=1) #[d]
        
        Hz  = torch.func.vmap(lambda xi, zi: torch.func.jacrev(lambda x_: g_single(x_, zi))(xi)
        )(x_dummy, z).view(-1, self.d, self.d) #[n, d, d]
        H_hat = (z*Hz).mean(dim=0) # [d, d] estimated Hessian of f_k wrt x_i, x_i

        return observed_deriv, H_hat

    def cycle_loss(self, W1: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        """Calculate the DAGMA constraint function

        Args:
            W (torch.Tensor): adjacency matrix
            s (float, optional): hyperparameter for the DAGMA constraint,
                can be any positive number. Defaults to 1.0.

        Returns:
            torch.Tensor: constraint
        """
        cycle_loss = -torch.slogdet(s * self.I - W1 * W1)[1] + self.d * np.log(s)

        return cycle_loss
    
    def ancestrality_loss(self, W1: torch.tensor, W2: torch.tensor)-> torch.Tensor:
        """
        Compute the loss due to violations of ancestrality in the induced ADMG of W1, W2.

        :param W1: numpy matrix for directed edge coefficients.
        :param W2: numpy matrix for bidirected edge coefficients.
        :return: float corresponding to penalty on violations of ancestrality.
        """
        d = len(W1)
        W1_pos = W1*W1
        W2_pos = W2*W2
        W1k = torch.eye(d)
        M = torch.eye(d)
        for k in range(1, d):
            W1k = W1k@W1_pos
            # M += comb(d, k) * (1 ** k) * W1k (typical binoimial)
            M += 1.0/math.factorial(k) * W1k #(special scaling)

        return torch.sum(M*W2_pos)
    
    def h_func(self, W1: torch.tensor, W2: torch.tensor, s: float = 1.0) -> torch.Tensor:

        cycle_loss= self.cycle_loss(W1, s)
        ancestrality_loss = self.ancestrality_loss(W1, W2)
        return cycle_loss+ancestrality_loss


    def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
        """Gets the L1 regularization

        Args:
            observed_derivs (torch.Tensor): the batched Jacobian matrix

        Returns:
            torch.Tensor: _description_
        """
        return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))
    

################

def hess_row_sumsq_vec(f, x, n_v=1, n_z=1, per_output_idx=None):
    """
    Estimate ∑_k || H^{(k)}_{i·} ||^2 for each input feature i.
    f: (B,d)->(B,p) model
    x: (B,d) with requires_grad=True
    n_v: # of output-probes v (random if per_output_idx is None)
    n_z: # of Hutchinson probes z per v
    per_output_idx: optional list/1D tensor of output indices to target strictly
    returns: curv_est (d,), sens_est (d,)   # curvature & sensitivity gates
    """
    B, d = x.shape
    with torch.enable_grad():
        curv_accum = torch.zeros(d, device=x.device)
        sens_accum = torch.zeros(d, device=x.device)

        # precompute f(x) once per v if you like; simplest is to recompute each loop
        n_v_eff = len(per_output_idx) if per_output_idx is not None else n_v

        for t in range(n_v_eff):
            fx = f(x)                                  # (n,d)
            p = fx.shape[1]

            if per_output_idx is not None:
                k = int(per_output_idx[t])
                v = torch.zeros_like(fx) # (n,d)
                v[:, k] = 1.0                          # strict: pick one output
            else:
                # random probe over outputs (Rademacher)
                v = (torch.randint_like(fx, 0, 2).float() * 2 - 1)  # (n,d)

            # Scalarize: y = <v, f(x)> averaged over batch, i.e., generate v per sample and do v^T f(x), then average over n
            y = (fx * v).sum() / B

            # First derivative wrt inputs
            g = torch.autograd.grad(y, x, create_graph=True)[0]   # (n,d)

            # Sensitivity gate E_v[(∂s_v/∂x_i)^2] ≈ ∑_k (∂f_k/∂x_i)^2
            sens_accum += (g**2).mean(dim=0)

            # Curvature via Hutchinson: Hz = ∇_x( g·z )
            curv_this_v = torch.zeros(d, device=x.device)
            for _ in range(n_z):
                z = (torch.randint_like(x, 0, 2).float() * 2 - 1) # (n,d)
                hz = torch.autograd.grad((g * z).sum(), x, create_graph=False)[0] # (d, d), hz[b,:] = (1/B)·(H_{s_{v_b}} z_b)
                curv_this_v += (hz**2).mean(dim=0).detach() # (d, 1)

            curv_accum += curv_this_v / n_z

        curv_est = curv_accum / n_v_eff
        sens_est = sens_accum / n_v_eff
        return curv_est, sens_est


def nonlin_regularizer_vec(f, x, m=1e-6, lam=1e-3, n_v=1, n_z=1, per_output_idx=None):
    """
    Option-A penalty for vector outputs:
    penalize nonzero sensitivity with near-zero curvature along each input dim.
    """
    curv, sens = hess_row_sumsq_vec(f, x, n_v=n_v, n_z=n_z, per_output_idx=per_output_idx)
    reg = (sens * torch.relu(m - curv)).sum()
    return lam * reg

