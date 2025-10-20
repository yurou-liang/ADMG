import nonlinear_dce
import nonlinear
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from locally_connected import LocallyConnected
import json
import argparse
import random
from torch.func import jacrev, vmap

def generate_ancestral_admg(d, p_dir=0.4, p_bidir=0.3, seed=None):
    """
    Generate an ancestral ADMG: {node: {'parents': [], 'spouses': []}}
    where spouses = bidirected connections (↔).
    p_dir: probability of a directed edge
    p_bidir: probability of a bidirected edge (latent confounding)
    admg: dict mapping each node to parents and spouses
    A_dir: directed adjacency matrix
    A_bidir: bidirected adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # --- Step 1: generate a DAG (acyclic directed structure)
    A_dir = np.triu((np.random.rand(d, d) < p_dir).astype(int), 1) # generate random 0/1 matrix with edge prob p_dir, keep only upper triangular part for acyclicty
    dag = {j: list(np.where(A_dir[:, j] == 1)[0]) for j in range(d)} # for each child j, take the indices i where A_dir[i, j] == 1 (i.e., it's parents)

    # --- Step 2: precompute ancestors of each node (for the ancestral constraint)
    ancestors = {j: set() for j in range(d)}
    for j in range(d):
        stack = list(dag[j])
        while stack:
            parent = stack.pop()
            ancestors[j].add(parent)
            stack.extend(dag[parent])  # recursively include higher ancestors

    # --- Step 3: generate bidirected edges that respect ancestrality
    A_bidir = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < p_bidir:
                # Check ancestral condition: i not ancestor of j, j not ancestor of i
                if i not in ancestors[j] and j not in ancestors[i]:
                    A_bidir[i, j] = A_bidir[j, i] = 1  # add bidirected edge

    # --- Step 4: return graph as dict
    admg = {
        j: {
            "parents": [int(i) for i in np.where(A_dir[:, j] == 1)[0]],
            "spouses": [int(i) for i in np.where(A_bidir[j, :] == 1)[0]],
        }
        for j in range(d)
    }
    return admg, A_dir, A_bidir

def generate_layers(d, dims, admg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    bias=True
    fc1 = nn.Linear(d, d * dims[1], bias=bias) # [d * dims[1], d]
    # self.fc1.weight.bounds = self._bounds()
    mask = torch.ones(d * dims[1], d)

    for j in range(d):
        allowed_parents = admg[j]['parents']
        not_parents = [p for p in range(d) if p not in allowed_parents]
        mask[j * dims[1]:(j + 1) * dims[1], not_parents] = 0.0

    layers = []
    for l in range(len(dims) - 2):
        layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
    fc2 = nn.ModuleList(layers)
    return fc1, fc2, mask

def scale_weights(model, factor=10):
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim > 1:  # Skip bias
                param.mul_(factor)

def forward(dims, fc1, fc2, mask, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the sigmoidal feedforward NN

    Args:
        x (torch.Tensor): input

    Returns:
        torch.Tensor: output
    """
    # x = self.fc1(x) # [n, self.d * dims[1]]
    weight = fc1.weight*mask #[d * dims[1], d]
    x = x@(weight.T) 
    if fc1.bias is not None:
        x = x + fc1.bias.unsqueeze(0)

    x = x.view(-1, dims[0], dims[1]) # [n, d, self.dims[1]]

    # self.activation = nn.SiLU()
    activation = nn.Sigmoid()

    for fc in fc2:
        # x = torch.sigmoid(x)
        x = activation(x)
        x = fc(x) # [n, d, self.dims[2]]

    x = x.squeeze(dim=2) #[n, d]

    return x

def generate_covariance(A_bidir, low=0.4, high=0.8, seed=None):
    """
    Generate a sparse, symmetric positive-definite covariance matrix
    whose sparsity follows A_bidir (1=nonzero).
    Correlations for nonzero entries are strong (0.4–0.8 by default).
    """
    if seed is not None:
        np.random.seed(seed)
    d = A_bidir.shape[0]

    # Step 1: random strong correlations for existing edges
    R = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            if A_bidir[i, j]:
                val = np.random.uniform(low, high)
                # Randomly flip sign for variety (optional)
                if np.random.rand() < 0.5:
                    val = -val
                R[i, j] = R[j, i] = val

    # Step 2: set diagonal to 1 temporarily
    np.fill_diagonal(R, 1.0)

    # Step 3: ensure positive definiteness
    # If smallest eigenvalue < 0, shift diagonal until PD
    eigvals = np.linalg.eigvalsh(R)
    if np.min(eigvals) <= 0:
        shift = abs(np.min(eigvals)) + 0.05  # small safety margin
        R += shift * np.eye(d)

    # Step 4: rescale diagonals to 1 again (optional normalization)
    D_inv = np.diag(1 / np.sqrt(np.diag(R)))
    Sigma = D_inv @ R @ D_inv

    # Final check
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.min(eigvals) <= 0:
        # small diagonal correction if needed
        Sigma += (abs(np.min(eigvals)) + 1e-6) * np.eye(d)

    return Sigma

def f(x_single):
    # expects x_single: shape [d]
    return forward(dims, fc1, fc2, mask, x_single.unsqueeze(0)).squeeze(0)

def reverse_SPDLogCholesky(Sigma: torch.tensor)-> torch.Tensor:
    """
    Reverse the LogCholesky decomposition that map the SPD Sigma matrix to the matrix M.
    """
    # Compute the Cholesky decomposition
    Sigma = torch.tensor(Sigma)
    L = torch.linalg.cholesky(Sigma)
    # Take strictly lower triangular matrix
    M_strict = L.tril(diagonal=-1)
    # Take the logarithm of the diagonal
    D = torch.diag(torch.log(L.diag()))
    # Return the log-Cholesky parametrization
    M = M_strict + D
    return M


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Comparison between RKHADagma and NOTEARS',)

    parser.add_argument('-d', '--num_nodes', dest='d', default=3, type=int)
    parser.add_argument('-s', '--seed', dest='s',  default=42, type=int)
    parser.add_argument('-T', '--num_iterations', dest='T', default=5, type=int)
    parser.add_argument('-gamma', default=10, type=int)

    args = parser.parse_args()
    torch.set_default_dtype(torch.double)
    np.random.seed(args.s)

    print(f'>>> Generating Data with MLP <<<')

    admg, A_dir, A_bidir = generate_ancestral_admg(args.d, p_dir=0.4, p_bidir=0.3, seed=args.s)
    n_samples = 1000 
    dims=[args.d, 10, 1]
    True_Sigma = generate_covariance(A_bidir, seed=args.s)
    epsilon = np.random.multivariate_normal([0] * args.d, True_Sigma, size=n_samples)
    epsilon = torch.tensor(epsilon)
    fc1, fc2, mask = generate_layers(args.d, dims, admg, seed = args.s)
    scale_weights(fc1, factor=10)
    for layer in fc2:
        scale_weights(layer, factor=10)
    X = forward(dims, fc1, fc2, mask, epsilon)
    X_true = (X+epsilon).detach()

    J = vmap(jacrev(f))(X_true)    # shape [n_samples, d, d]
    W_true = torch.sqrt(torch.mean(J ** 2, axis=0).T)
    print("W_true: ", W_true)


    M_truth = reverse_SPDLogCholesky(True_Sigma)

    # print(f'>>> DAGMA Init <<<')

    # eq_model = nonlinear.DagmaMLP(
    # dims=[args.d, 10, 1], bias=True, dtype=torch.double)
    # model = nonlinear.DagmaNonlinear(
    #     eq_model, dtype=torch.double, use_mse_loss=True)

    # W_est_dagma = model.fit(X_true, lambda1=2e-2, lambda2=0.005,
    #                         T=1, lr=2e-4, w_threshold=0.3, mu_init=0.1, warm_iter=70000, max_iter=80000)

    # # Use DAGMA weights as initial weights for DAGMA-DCE
    # fc1_weight_DAGMA = eq_model.fc1.weight
    # fc1_bias_DAGMA = eq_model.fc1.bias
    # fc2_weight_DAGMA = eq_model.fc2[0].weight
    # fc2_bias_DAGMA = eq_model.fc2[0].bias

    # eq_model = nonlinear_dce.DagmaMLP_DCE(
    #     dims=[args.d, 10, 1], bias=True)
    # model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True)
    # eq_model.fc1.weight = fc1_weight_DAGMA
    # eq_model.fc1.bias = fc1_bias_DAGMA
    # eq_model.fc2[0].weight = fc2_weight_DAGMA
    # eq_model.fc2[0].bias = fc2_bias_DAGMA

    # W_est_DAGMA, W2_DAGMA, x_est_DAGMA = model.fit(X_true, lambda1=3.5e-2, lambda2=5e-3,
    #                                 lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    
    	
    # _, observed_derivs = eq_model.get_graph(X_true)
    # observed_derivs_mean = observed_derivs.mean(dim = 0)
    # observed_hess_DAGMA = eq_model.exact_hessian_diag_avg(X_true)
    # Sigma_est_DAGMA = eq_model.get_Sigma()
    # mle_loss_DAGMA = model.mle_loss(x_est_DAGMA, X_true, Sigma_est_DAGMA)
    # h_val_DAGMA = eq_model.h_func(W_est_DAGMA, W2_DAGMA)
    # nonlinear_reg_DAGMA = eq_model.get_nonlinear_reg(observed_derivs_mean, observed_hess_DAGMA)
    
    # print(f'>>> random Init <<<')

    # eq_model = nonlinear.DagmaMLP(
    # dims=[args.d, 10, 1], bias=True, dtype=torch.double)
    # model = nonlinear.DagmaNonlinear(
    #     eq_model, dtype=torch.double, use_mse_loss=True)
    # W_est_random, W2_random, x_est_random = model.fit(X_true, lambda1=3.5e-2, lambda2=5e-3,
    #                                 lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    # _, observed_derivs = eq_model.get_graph(X_true)
    # observed_derivs_mean = observed_derivs.mean(dim = 0)
    # observed_hess_random = eq_model.exact_hessian_diag_avg(X_true)
    # Sigma_est_random = eq_model.get_Sigma()
    # mle_loss_random = model.mle_loss(x_est_random, X_true, Sigma_est_random)
    # h_val_random = eq_model.h_func(W_est_random, W2_random)
    # nonlinear_reg_random = eq_model.get_nonlinear_reg(observed_derivs_mean, observed_hess_random)
    
    print(f'>>> Truth Init <<<')

    fc1_weight_truth = fc1.weight*mask
    fc1_bias_truth = fc1.bias
    fc2_weight_truth = fc2[0].weight
    fc2_bias_truth = fc2[0].bias

    eq_model = nonlinear_dce.DagmaMLP_DCE(
        dims=[args.d, 10, 1], bias=True)
    model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True)
    eq_model.fc1.weight = nn.Parameter(fc1_weight_truth)
    eq_model.fc1.bias = fc1_bias_truth
    eq_model.fc2[0].weight = fc2_weight_truth
    eq_model.fc2[0].bias = fc2_bias_truth
    eq_model.M = nn.Parameter(M_truth.clone())
    W_est_truth, W2_truth, x_est_truth = model.fit(X_true, lambda1=3.5e-2, lambda2=5e-3,
                                lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    _, observed_derivs = eq_model.get_graph(X_true)
    observed_derivs_mean = observed_derivs.mean(dim = 0)
    observed_hess_truth = eq_model.exact_hessian_diag_avg(X_true)
    Sigma_est_truth = eq_model.get_Sigma()
    mle_loss_truth = model.mle_loss(x_est_truth, X_true, Sigma_est_truth)
    h_val_truth = eq_model.h_func(W_est_truth, W2_truth)
    nonlinear_reg_truth = eq_model.get_nonlinear_reg(observed_derivs_mean, observed_hess_truth)

    filename = f'result_d{args.d}_seed{args.s}'
    results = {
    'admg': admg,
    'X_true': X_true.detach().cpu().numpy().tolist(),
    'W_true': W_true.detach().cpu().numpy().tolist(),
    "True_Sigma": True_Sigma.tolist(),
    "fc1_weight_truth": fc1_weight_truth.detach().cpu().tolist(),
    "fc1_bias_truth": fc1_bias_truth.detach().cpu().tolist(),
    "fc2_weight_truth": fc2_weight_truth.detach().cpu().tolist(),
    "fc2_bias_truth": fc2_bias_truth.detach().cpu().tolist(),
    "M_truth": M_truth.detach().cpu().tolist(),
    'h_val_truth': h_val_truth.item(),
    'mle_loss_truth': mle_loss_truth.detach().cpu().tolist(),
    'W_est_truth': W_est_truth.detach().cpu().tolist(),
    'Sigma_est_truth': Sigma_est_truth.detach().cpu().tolist(),
    # 'h_val_DAGMA': h_val_DAGMA.item(),
    # 'mle_loss_DAGMA': mle_loss_DAGMA.detach().cpu().tolist(),
    # 'W_est_DAGMA': W_est_DAGMA.detach().cpu().tolist(),
    # 'Sigma_est_DAGMA': Sigma_est_DAGMA.detach().cpu().tolist(),
    # "fc1_weight_DAGMA": fc1_weight_DAGMA.detach().cpu().tolist(),
    # "fc1_bias_DAGMA": fc1_bias_DAGMA.detach().cpu().tolist(),
    # "fc2_weight_DAGMA": fc2_weight_DAGMA.detach().cpu().tolist(),
    # "fc2_bias_DAGMA": fc2_bias_DAGMA.detach().cpu().tolist(),
    # 'h_val_random': h_val_random.item(),
    # 'mle_loss_random': mle_loss_random.detach().cpu().tolist(),
    # 'W_est_random': W_est_random.detach().cpu().tolist(),
    # 'Sigma_est_random': Sigma_est_random.detach().cpu().tolist()
    }

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

    

    



