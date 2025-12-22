# Modified from https://github.com/kevinsbello/dagma/blob/main/src/dagma/nonlinear.py
# Modifications Copyright (C) 2023 Dan Waxman

import copy
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
import torch.nn.functional as F


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
        Sigma = Sigma
        tmp = torch.linalg.solve(Sigma, (target - output).T)
        mle = torch.trace((target - output)@tmp)/n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle
    
    def stable_mle_loss(self, output: torch.Tensor, target: torch.Tensor, Sigma: torch.Tensor, jitter: float = 1e-4):
        E = target - output
        n, d = E.shape
        I = torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
        Sigma_j = Sigma + jitter * I

        try:
            L = torch.linalg.cholesky(Sigma_j)
        except RuntimeError:
            # fallback: add more jitter dynamically
            Sigma_j = Sigma_j + 1e-2 * I
            L = torch.linalg.cholesky(Sigma_j)

        tmp = torch.cholesky_solve(E.T, L)
        quad = torch.sum(E.T * tmp) / n
        logdet = 2 * torch.sum(torch.log(torch.diag(L)))
        return quad + logdet


    def minimize(
        self,
        max_iter: int,
        indi: float,
        indi_h: float,
        lr: float,
        lambda1: float,
        lambda_corr: float,
        lambda_nl: float,
        lambda2: float,
        mu: float,
        s: float,
        pbar: tqdm,
        lr_decay: bool = False,
        checkpoint: int = 1000,
        tol: float = 1e-3,
        freeze_Sigma: bool = False
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
        # optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr=lr,
        #     betas=(0.99, 0.999),
        #     weight_decay=mu * lambda2,
        # )

        params_f = []
        params_sigma = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name == "M":
                params_sigma.append(p)
            else:
                params_f.append(p)

        lr_sigma = lr * 0.3  # you can try 0.1, 0.03, etc.
        optimizer = optim.Adam(
            [
                {"params": params_f,     "lr": lr,        "weight_decay": mu * lambda2},
                {"params": params_sigma, "lr": lr_sigma,  "weight_decay": mu * lambda2},
            ],
            betas=(0.99, 0.999),
        )
        
        obj_prev = 1e16

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8 if lr_decay else 1.0
        )

        # if freeze_Sigma:
        #     self.model.M.requires_grad_(False)
        # else:
        #     self.model.M.requires_grad_(True)

        for i in range(max_iter):
            optimizer.zero_grad()
            # if i == 0:
            #     self.model.M.requires_grad_(True)
            if i == 0:
                X_hat = self.model(self.X)
                Sigma = self.model.get_Sigma()
                score = self.loss(X_hat, self.X, Sigma)
                obj = score
                Wii = torch.diag(torch.diag(Sigma))
                W2 = Sigma - Wii
                W_current, observed_derivs = self.model.get_graph(self.X)
                observed_derivs_mean = observed_derivs.abs().mean(dim = 0)
                observed_hess = self.model.exact_hessian_diag_avg(self.X)
                h_val = self.model.h_func(W_current, W2, s)
                nonlinear_reg = self.model.get_nonlinear_reg(observed_derivs_mean, observed_hess)
                # print("Sigma: ", Sigma)
                # print("obj: ", obj)
                # print("mle loss: ", score)
                # print("h_val: ", h_val)
                # print("nonlinear_reg: ", nonlinear_reg)
                # print("observed_derivs: ", observed_derivs_mean)
                # print("observed_hess: ", observed_hess)
                # print("mu: ", mu)
                # print("W_current: ", W_current)

            else:
                W_current, observed_derivs = self.model.get_graph(self.X)
                observed_derivs_mean = observed_derivs.abs().mean(dim = 0)
                observed_hess = self.model.exact_hessian_diag_avg(self.X)
                Sigma = self.model.get_Sigma()
                ##### new test
                # Sigma = torch.eye(self.X.shape[1])
                Wii = torch.diag(torch.diag(Sigma))
                W2 = Sigma - Wii
                h_val = self.model.h_func(W_current, W2, s)

                if h_val.item() < 0:
                    return False

                X_hat = self.model(self.X)
                score = self.loss(X_hat, self.X, Sigma)

                l1_reg = lambda1 * self.model.get_l1_reg(observed_derivs)
                nonlinear_reg = self.model.get_nonlinear_reg(observed_derivs_mean, observed_hess)
                # W2_reg = 3*lambda1 * self.model.get_W2_reg(W2)
                corr_reg = self.model.get_W2_reg(W2, lambda_corr)
                sigma_reg = self.model.sigma_rel_reg(Sigma, lambda_diag=15*lambda1)

                obj = mu * (score + indi*(l1_reg + corr_reg + lambda_nl*nonlinear_reg)) + indi_h*h_val
                

                # obj = mu * (score + l1_reg) + h_val

                if i % 1000 == 0:
                    print("Sigma: ", Sigma)
                    print("W_current: ", W_current)
                    print("W2: ", W2)
                    print("obj: ", obj)
                    print("mle loss: ", score)
                    print("h_val: ", h_val)
                    print("nonlinear_reg: ", nonlinear_reg)
                    print("observed_derivs: ", observed_derivs_mean)
                    # print("W_current.T: ", W_current.T)
                    print("observed_hess: ", observed_hess)
                    print("mu: ", mu)

            obj.backward()
            # Clip gradients to avoid a big jump
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            optimizer.step()

            # Immediately check params
            with torch.no_grad():
                bad = False
                for name, p in self.model.named_parameters():
                    if p is None: 
                        continue
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print(f"⚠️ NaN/Inf in parameter {name}")
                        bad = True
                if bad:
                    return False  # trigger rollback / lr reduce
            

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
        lambda_corr: float = 0.02,
        lambda_nl: float = 8.0,
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
        indi = 1.0
        with tqdm(total=(T - 1) * warm_iter + max_iter, disable=disable_pbar) as pbar:
            for i in range(int(T)):
                success, s_cur = False, s
                lr_decay = False

                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                # indi_h =0.05 if i == T-1 else 1.0 
                # indi_i = 0.0 if i == T - 1 else indi
                indi_h = 1.0
                indi_i = 1.0

                model_copy = copy.deepcopy(self.model)
                # freeze_Sigma = True if i == 0 else False
                while success is False:
                    success = self.minimize(
                        inner_iter,
                        indi_i,
                        indi_h,
                        lr,
                        lambda1,
                        lambda_corr,
                        lambda_nl,
                        lambda2,
                        mu,
                        s_cur,
                        lr_decay=lr_decay,
                        pbar=pbar,
                        freeze_Sigma=False
                    )

                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break  # lr is too small

                    mu *= mu_factor
                    # indi *= 0.7

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
    d,_ = M_strict.shape
    # Make matrix with exponentiated diagonal
    D = M.diag()
    # Make the Cholesky decomposition matrix
    L = M_strict + torch.diag(torch.exp(D))
    # Invert the Cholesky decomposition
    Sigma = torch.matmul(L, L.t())
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

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", mask)

        # Zero gradients for masked weights during backprop
        self.weight.register_hook(lambda grad: grad * self.mask)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)
    
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

        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias) # [d * dims[1], d]
        # self.fc1.weight.bounds = self._bounds()
        self.mask = torch.ones(self.d * dims[1], self.d)

        for j in range(self.d):
            self.mask[j * dims[1]:(j + 1) * dims[1], j] = 0.0+1e-6

        # self.fc1 = MaskedLinear(self.d, self.d * dims[1], mask, bias=bias)
        # self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)

        # with torch.no_grad():
        #     self.fc1.weight *= mask

        # nn.init.zeros_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc1.weight.data *= self.fc1.mask   # enforce mask after init
        nn.init.zeros_(self.fc1.bias)

        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
            # Each dimension d has a separate weight to learn

        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sigmoidal feedforward NN

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        # x = self.fc1(x) # [n, self.d * dims[1]]
        weight = self.fc1.weight*self.mask #[d * dims[1], d]
        x = x@(weight.T) 
        if self.fc1.bias is not None:
            x = x + self.fc1.bias.unsqueeze(0)

        x = x.view(-1, self.dims[0], self.dims[1]) # [n, d, self.dims[1]]

        # self.activation = nn.SiLU()
        self.activation = nn.Sigmoid()

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
        if torch.isnan(x).any():
            print("NaN in input X before jacobian!")
        y = self.forward(x)
        if torch.isnan(y).any():
            print("NaN in forward output!")
        x_dummy = x.detach().requires_grad_()

        observed_deriv = torch.func.vmap(torch.func.jacrev(self.forward))(x_dummy).view(
            -1, self.d, self.d
        )#[n, d, d], observed_deriv[i, j, k]=for ith sample, derivative of f_j wrt x_k

        # Adjacency matrix is RMS Jacobian
        W = torch.sqrt(torch.mean(observed_deriv**2, axis=0).T) #[d, d]

        if torch.isnan(observed_deriv).any():
            print("NaN in jacobian!")

        return W, observed_deriv
    
    def exact_hessian_diag_avg(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """
        Exact per-output Hessian diagonals, averaged over a large batch.
        model.forward: [B, d] -> [B, d]
        x: [n, d]
        returns: [d, d] where out[k, i] = average over samples of ∂^2 f_k / ∂x_i^2
        """
        device = x.device
        n, d = x.shape
        out = torch.zeros(d, d, device=device)
        total = 0

        # per-sample wrapper so higher-order AD sees [d] -> [d]
        def f_single(x1):  # x1: [d] -> [d]
            return self.forward(x1.unsqueeze(0)).squeeze(0)

        # scalarization: s(x; u) = <u, f(x)>, Hess_x s with u = e_k gives Hessian of f_k
        def s(x1, u1):
            return (f_single(x1) * u1).sum()

        # Hessian wrt x only (treat u as constant)
        hess_x = torch.func.hessian(s, argnums=0)  # (x1, u1) -> [d, d]
        I = torch.eye(d, device=device)            # unit vectors e_k as rows

        # For one sample: stack all outputs' Hessians -> [d(outputs), d, d]
        def hess_all_outputs_for_sample(xi):
            return torch.func.vmap(lambda u: hess_x(xi, u), in_dims=0)(I)  # [d, d, d]

        with torch.no_grad():  # we only need numbers, not a backward graph
            for start in range(0, n, batch_size):
                xb = x[start:start+batch_size].detach().to(device).requires_grad_(True)
                # Map over the minibatch: [B, d, d, d]
                H = torch.func.vmap(hess_all_outputs_for_sample, in_dims=0)(xb)
                # Take diagonal over last two dims -> [B, d, d]
                Hdiag = torch.diagonal(H, dim1=-2, dim2=-1).contiguous()
                out += Hdiag.abs().sum(dim=0)  # accumulate sum over this minibatch
                total += Hdiag.size(0)

        return out / total  # [d, d]

    def cycle_loss(self, W1: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        """Calculate the DAGMA constraint function

        Args:
            W (torch.Tensor): adjacency matrix
            s (float, optional): hyperparameter for the DAGMA constraint,
                can be any positive number. Defaults to 1.0.

        Returns:
            torch.Tensor: constraint
        """
        # A = s * self.I - W1 * W1
        # print("A min diag:", A.diag().min().item())
        # print("A max diag:", A.diag().max().item())
        # print("||A||_F =", torch.linalg.norm(A).item())

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
    
    def h_func(self, W1: torch.tensor, W2: torch.tensor, s: float = 32.0) -> torch.Tensor:

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
    
    def get_nonlinear_reg(self, observed_derivs, observed_hess, m=1e-1):
        # constants on the right device/dtype
        m_t = torch.as_tensor(m, device=observed_hess.device, dtype=observed_hess.dtype)

        # encourage |H| >= m where |J| is large
        # detach H so we don't backprop through second→third order
        gap = torch.clamp_min(m_t - observed_hess.abs(), 0.0)  # [d, d]

        # broadcast over batch n; penalty per (sample, j, k)
        penalty = observed_derivs.abs() * gap  # [n, d, d]

        return penalty.sum()

    # def get_nonlinear_reg(self, observed_derivs, observed_hess, m=1e-1, tau=1e-2):
    #     # mask: where the derivative is meaningfully nonzero (edge exists)
    #     # use J only as a mask, without gradient
    #     J_mean = observed_derivs.abs().mean(dim=0).detach()   # [d, d]
    #     mask = (J_mean > tau).float()                         # 1 where edge is active

    #     m_t = torch.as_tensor(m, device=observed_hess.device, dtype=observed_hess.dtype)

    #     # DO NOT detach H here: we want gradient to push curvature up
    #     gap = torch.clamp_min(m_t - observed_hess.abs(), 0.0)  # [d, d]

    #     # penalty: where mask=1 and curvature is too small
    #     penalty = (mask * gap).sum()

    #     return penalty


    def get_W2_reg(self, W2: torch.Tensor, lambda_corr: float) -> torch.Tensor:

        return lambda_corr*0.5*torch.sum(W2.abs())
    
    def get_corr_reg(self, Sigma: torch.Tensor, lambda_corr: float) -> torch.Tensor:
        eps = 1e-12
        d = Sigma.size(0)
        Dinv = torch.diag(1.0 / torch.sqrt(torch.diag(Sigma) + eps))
        Corr = Dinv @ Sigma @ Dinv
        off_corr = Corr - torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
        corr_sparsity = lambda_corr * off_corr.abs().sum()
        return corr_sparsity

    def sigma_rel_reg(self, Sigma: torch.Tensor, lambda_diag: float = 1e-3) -> torch.Tensor:
        """
        Penalize relative spread of log-variances: encourages diag(Sigma)
        to be on a similar scale across nodes, without fixing it to 1.
        """
        diag = torch.diag(Sigma)
        log_diag = torch.log(diag + 1e-12)
        mean_log = log_diag.mean().detach()
        return lambda_diag * ((log_diag - mean_log) ** 2).sum()

