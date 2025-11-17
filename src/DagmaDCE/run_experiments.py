import nonlinear_dce
import nonlinear
import torch
import numpy as np
import networkx as nx
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
        mask[j * dims[1]:(j + 1) * dims[1], not_parents] = 0.0 + 1e-6

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
    # Sigma = torch.tensor(Sigma)
    L = torch.linalg.cholesky(Sigma)
    # Take strictly lower triangular matrix
    M_strict = L.tril(diagonal=-1)
    # Take the logarithm of the diagonal
    D = torch.diag(torch.log(L.diag()))
    # Return the log-Cholesky parametrization
    M = M_strict + D
    return M

def generate_from_epsilon(dims, epsilon, fc1, fc2, mask, parents):
    """
    Generate data x from epsilon according to x = f(x_parents) + epsilon.
    
    Args:
        epsilon: [n, d] tensor of Gaussian noise
        fc1, fc2, mask: network parameters
        parents: dict {j: [parents_of_j]}
    Returns:
        x: [n, d] tensor of generated variables
    """
    n, d = epsilon.shape
    X = torch.zeros_like(epsilon)

    # Determine topological order (if not already given)
    order = list(parents.keys())  # assume already topologically sorted

    for j in order:
        if len(parents[j]) == 0:
            # root node: only noise
            X[:, j] = epsilon[:, j]
        else:
            # prepare partial input with parents filled
            x_partial = torch.zeros(n, d, dtype=epsilon.dtype)
            for p in parents[j]:
                x_partial[:, p] = X[:, p]

            # compute all f_j(x) in parallel, then pick j-th column
            f_out = forward(dims, fc1, fc2, mask, x_partial)  # [n, d]
            X[:, j] = f_out[:, j] + epsilon[:, j]

    return X

def mle_loss(output: torch.Tensor, target: torch.Tensor, Sigma: torch.Tensor):
        """Computes the MLE loss 1/n*Tr((X-X_est)Sigma^{-1}(X-X_est)^T)"""
        n, d = target.shape
        tmp = torch.linalg.solve(Sigma, (target - output).T)
        mle = torch.trace((target - output)@tmp)/n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle


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
    print("admg: ", admg)
    parents = {j: admg[j]['parents'] for j in admg}
    G = nx.DiGraph()
    G.add_nodes_from(parents.keys())
    for j, pa in parents.items():
        G.add_edges_from((p, j) for p in pa)
    order = list(nx.topological_sort(G))

    n_samples = 1000 
    dims=[args.d, 10, 1]
    Sigma_truth = generate_covariance(A_bidir, seed=10)
    epsilon = np.random.multivariate_normal([0] * args.d, Sigma_truth, size=n_samples)
    Sigma_truth = torch.tensor(Sigma_truth)
    epsilon = torch.tensor(epsilon)
    fc1, fc2, mask = generate_layers(args.d, dims, admg, seed = 13)
    scale_weights(fc1, factor=10)
    for layer in fc2:
        scale_weights(layer, factor=20)
    X_truth = generate_from_epsilon(dims, epsilon, fc1, fc2, mask, parents).detach()
    X = X_truth - epsilon

    J = vmap(jacrev(f))(X_truth)    # shape [n_samples, d, d]
    W_truth = torch.sqrt(torch.mean(J ** 2, axis=0).T)
    
    mle_loss_truth = mle_loss(X, X_truth, Sigma_truth)

    M_truth = reverse_SPDLogCholesky(Sigma_truth)

    print(f'>>> DAGMA Init with h <<<')

    eq_model = nonlinear.DagmaMLP(
    dims=[args.d, 10, 1], bias=True, dtype=torch.double)
    model = nonlinear.DagmaNonlinear(
        eq_model, dtype=torch.double, use_mse_loss=True)

    W_est_dagma_with_h = model.fit(X_truth, lambda1=2e-2, lambda2=0.005,
                            T=1, lr=2e-4, w_threshold=0.3, mu_init=1, warm_iter=70000, max_iter=80000, consider_h=False)

    # Use DAGMA weights as initial weights for DAGMA-DCE
    fc1_weight_DAGMA_with_h = eq_model.fc1.weight
    fc1_bias_DAGMA_with_h = eq_model.fc1.bias
    fc2_weight_DAGMA_with_h = eq_model.fc2[0].weight
    fc2_bias_DAGMA_with_h= eq_model.fc2[0].bias

    eq_model = nonlinear_dce.DagmaMLP_DCE(
        dims=[args.d, 10, 1], bias=True)
    model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True)
    eq_model.fc1.weight = fc1_weight_DAGMA_with_h
    eq_model.fc1.bias = fc1_bias_DAGMA_with_h
    eq_model.fc2[0].weight = fc2_weight_DAGMA_with_h
    eq_model.fc2[0].bias = fc2_bias_DAGMA_with_h

    W_est_DAGMA_with_h, W2_DAGMA_with_h, x_est_DAGMA_with_h = model.fit(X_truth, lambda1=3.5e-2, lambda2=5e-3,
                                    lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    
    fc1_weight_DAGMA_with_h_end = eq_model.fc1.weight
    fc1_bias_DAGMA_with_h_end = eq_model.fc1.bias
    fc2_weight_DAGMA_with_h_end = eq_model.fc2[0].weight
    fc2_bias_DAGMA_with_h_end = eq_model.fc2[0].bias
    
    	
    _, observed_derivs = eq_model.get_graph(X_truth)
    observed_derivs_mean_with_h = observed_derivs.mean(dim = 0)
    observed_hess_DAGMA_with_h = eq_model.exact_hessian_diag_avg(X_truth)
    Sigma_est_DAGMA_with_h = eq_model.get_Sigma()
    mle_loss_DAGMA_with_h = model.mle_loss(x_est_DAGMA_with_h, X_truth, Sigma_est_DAGMA_with_h)
    h_val_DAGMA_with_h = eq_model.h_func(W_est_DAGMA_with_h, W2_DAGMA_with_h)
    nonlinear_reg_DAGMA_with_h = eq_model.get_nonlinear_reg(observed_derivs_mean_with_h, observed_hess_DAGMA_with_h)

    # print(f'>>> DAGMA Init without h <<<')

    # eq_model = nonlinear.DagmaMLP(
    # dims=[args.d, 10, 1], bias=True, dtype=torch.double)
    # model = nonlinear.DagmaNonlinear(
    #     eq_model, dtype=torch.double, use_mse_loss=True)

    # W_est_dagma_without_h = model.fit(X_truth, lambda1=2e-2, lambda2=0.005,
    #                         T=1, lr=2e-4, w_threshold=0.3, mu_init=1, warm_iter=70000, max_iter=80000, consider_h=False)

    # # Use DAGMA weights as initial weights for DAGMA-DCE
    # fc1_weight_DAGMA_without_h = eq_model.fc1.weight
    # fc1_bias_DAGMA_without_h = eq_model.fc1.bias
    # fc2_weight_DAGMA_without_h = eq_model.fc2[0].weight
    # fc2_bias_DAGMA_without_h= eq_model.fc2[0].bias

    # eq_model = nonlinear_dce.DagmaMLP_DCE(
    #     dims=[args.d, 10, 1], bias=True)
    # model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True)
    # eq_model.fc1.weight = fc1_weight_DAGMA_without_h
    # eq_model.fc1.bias = fc1_bias_DAGMA_without_h
    # eq_model.fc2[0].weight = fc2_weight_DAGMA_without_h
    # eq_model.fc2[0].bias = fc2_bias_DAGMA_without_h

    # W_est_DAGMA_without_h, W2_DAGMA_without_h, x_est_DAGMA_without_h = model.fit(X_truth, lambda1=3.5e-2, lambda2=5e-3,
    #                                 lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    
    # fc1_weight_DAGMA_without_h_end = eq_model.fc1.weight
    # fc1_bias_DAGMA_without_h_end = eq_model.fc1.bias
    # fc2_weight_DAGMA_without_h_end = eq_model.fc2[0].weight
    # fc2_bias_DAGMA_without_h_end = eq_model.fc2[0].bias
    
    	
    # _, observed_derivs = eq_model.get_graph(X_truth)
    # observed_derivs_mean_without_h = observed_derivs.mean(dim = 0)
    # observed_hess_DAGMA_without_h = eq_model.exact_hessian_diag_avg(X_truth)
    # Sigma_est_DAGMA_without_h = eq_model.get_Sigma()
    # mle_loss_DAGMA_without_h = model.mle_loss(x_est_DAGMA_without_h, X_truth, Sigma_est_DAGMA_without_h)
    # h_val_DAGMA_without_h = eq_model.h_func(W_est_DAGMA_without_h, W2_DAGMA_without_h)
    # nonlinear_reg_DAGMA_without_h = eq_model.get_nonlinear_reg(observed_derivs_mean_without_h, observed_hess_DAGMA_without_h)
    
    # print(f'>>> Truth Init <<<')

    # fc1_weight_truth = fc1.weight*mask
    # fc1_bias_truth = fc1.bias
    # fc2_weight_truth = fc2[0].weight
    # fc2_bias_truth = fc2[0].bias

    # eq_model = nonlinear_dce.DagmaMLP_DCE(
    #     dims=[args.d, 10, 1], bias=True)
    # model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True)
    # eq_model.mask = mask.clone()
    # eq_model.fc1.weight = nn.Parameter(fc1_weight_truth)
    # eq_model.fc1.bias = fc1_bias_truth
    # eq_model.fc2[0].weight = fc2_weight_truth
    # eq_model.fc2[0].bias = fc2_bias_truth

    # with torch.no_grad():
    #     for j, pa in parents.items():
    #         if len(pa) == 0:
    #             start = j * dims[1]
    #             end = (j + 1) * dims[1]
    #             eq_model.fc1.bias[start:end] = 0.0 + 1e-6
    #             eq_model.fc2[0].bias[j] = 0.0+ 1e-6
    #             eq_model.fc2[0].weight[j, :, 0] = 0.0+ 1e-6

    # eq_model.M = nn.Parameter(M_truth.clone())

    # # Check forward equivalence
    # with torch.no_grad():
    #     X_hat = eq_model(X_truth)
    #     eps = X_truth - X_hat
    #     average_abs_epsilon = eps.abs().mean().item()
    #     print("mean |eps|:", average_abs_epsilon)
    #     Sigma_emp = (eps.T @ eps) / eps.size(0)
    #     diff_emp_true_Sigma = (Sigma_emp - torch.tensor(Sigma_truth, dtype=torch.double)).clone().detach().abs().max()
    #     print("‖Σ_emp − True_Sigma‖:", diff_emp_true_Sigma)

    # W_truth_start, observed_derivs = eq_model.get_graph(X_truth)
    # X_truth_hat_start = eq_model.forward(X_truth)
    # mle_loss_truth_start = mle_loss(X_truth_hat_start, X_truth, Sigma_truth)

    # W_est_truth, W2_truth, x_est_truth = model.fit(X_truth, lambda1=3.5e-2, lambda2=5e-3,
    #                             lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    
    # fc1_weight_truth_end = eq_model.fc1.weight
    # fc1_bias_truth_end = eq_model.fc1.bias
    # fc2_weight_truth_end = eq_model.fc2[0].weight
    # fc2_bias_truth_end = eq_model.fc2[0].bias
    
    # _, observed_derivs = eq_model.get_graph(X_truth)
    # observed_derivs_mean = observed_derivs.mean(dim = 0)
    # observed_hess_truth = eq_model.exact_hessian_diag_avg(X_truth)
    # Sigma_est_truth = eq_model.get_Sigma()
    # mle_loss_truth_end = model.mle_loss(x_est_truth, X_truth, Sigma_est_truth)
    # h_val_truth = eq_model.h_func(W_est_truth, W2_truth)
    # nonlinear_reg_truth = eq_model.get_nonlinear_reg(observed_derivs_mean, observed_hess_truth)

    filename = f'result_d{args.d}_seed{args.s}_modified'
    results = {
    'admg': admg,
    'X_truth': X_truth.detach().cpu().numpy().tolist(),
    'X': X.detach().cpu().numpy().tolist(),
    'epsilon': epsilon.detach().cpu().numpy().tolist(),
    'W_truth': W_truth.detach().cpu().numpy().tolist(),
    # 'W_start': W_truth_start.detach().cpu().numpy().tolist(),
    "Sigma_truth": Sigma_truth.tolist(),
    "M_truth": M_truth.detach().cpu().tolist(),
    # 'h_val_truth': h_val_truth.item(),
    'mle_loss_truth': mle_loss_truth.detach().cpu().tolist(),
    # 'mle_loss_truth_start': mle_loss_truth_start.detach().cpu().tolist(),
    # 'mle_loss_truth_end': mle_loss_truth_end.detach().cpu().tolist(),
    # 'W_est_truth': W_est_truth.detach().cpu().tolist(),
    # 'Sigma_est_truth': Sigma_est_truth.detach().cpu().tolist(),
    # "nonlinear_reg_truth": nonlinear_reg_truth.detach().cpu().tolist(),
    # "fc1_weight_truth_start": fc1_weight_truth.detach().cpu().tolist(),
    # "fc1_bias_truth_start": fc1_bias_truth.detach().cpu().tolist(),
    # "fc2_weight_truth_start": fc2_weight_truth.detach().cpu().tolist(),
    # "fc2_bias_truth_start": fc2_bias_truth.detach().cpu().tolist(),
    # "fc1_weight_truth_end": fc1_weight_truth_end.detach().cpu().tolist(),
    # "fc1_bias_truth_end": fc1_bias_truth_end.detach().cpu().tolist(),
    # "fc2_weight_truth_end": fc2_weight_truth_end.detach().cpu().tolist(),
    # "fc2_bias_truth_end": fc2_bias_truth_end.detach().cpu().tolist(),

    'mle_loss_DAGMA_with_h': mle_loss_DAGMA_with_h.detach().cpu().tolist(),
    'W_est_DAGMA_with_h': W_est_DAGMA_with_h.detach().cpu().tolist(),
    'Sigma_est_DAGMA_with_h': Sigma_est_DAGMA_with_h.detach().cpu().tolist(),
    "nonlinear_reg_DAGMA_with_h": nonlinear_reg_DAGMA_with_h.detach().cpu().tolist(),
    "fc1_weight_DAGMA_with_h_start": fc1_weight_DAGMA_with_h.detach().cpu().tolist(),
    "fc1_bias_DAGMA_with_h_start": fc1_bias_DAGMA_with_h.detach().cpu().tolist(),
    "fc2_weight_DAGMA_with_h_start": fc2_weight_DAGMA_with_h.detach().cpu().tolist(),
    "fc2_bias_DAGMA_with_h_start": fc2_bias_DAGMA_with_h.detach().cpu().tolist(),
    "fc1_weight_DAGMA_with_h_end": fc1_weight_DAGMA_with_h_end.detach().cpu().tolist(),
    "fc1_bias_DAGMA_with_h_end": fc1_bias_DAGMA_with_h_end.detach().cpu().tolist(),
    "fc2_weight_DAGMA_with_h_end": fc2_weight_DAGMA_with_h_end.detach().cpu().tolist(),
    "fc2_bias_DAGMA_with_h_end": fc2_bias_DAGMA_with_h_end.detach().cpu().tolist(),

    # 'mle_loss_DAGMA_without_h': mle_loss_DAGMA_without_h.detach().cpu().tolist(),
    # 'W_est_DAGMA_without_h': W_est_DAGMA_without_h.detach().cpu().tolist(),
    # 'Sigma_est_DAGMA_without_h': Sigma_est_DAGMA_without_h.detach().cpu().tolist(),
    # "nonlinear_reg_DAGMA_without_h": nonlinear_reg_DAGMA_without_h.detach().cpu().tolist(),
    # "fc1_weight_DAGMA_without_h_start": fc1_weight_DAGMA_without_h.detach().cpu().tolist(),
    # "fc1_bias_DAGMA_without_h_start": fc1_bias_DAGMA_without_h.detach().cpu().tolist(),
    # "fc2_weight_DAGMA_without_h_start": fc2_weight_DAGMA_without_h.detach().cpu().tolist(),
    # "fc2_bias_DAGMA_without_h_start": fc2_bias_DAGMA_without_h.detach().cpu().tolist(),
    # "fc1_weight_DAGMA_without_h_end": fc1_weight_DAGMA_without_h_end.detach().cpu().tolist(),
    # "fc1_bias_DAGMA_without_h_end": fc1_bias_DAGMA_without_h_end.detach().cpu().tolist(),
    # "fc2_weight_DAGMA_without_h_end": fc2_weight_DAGMA_without_h_end.detach().cpu().tolist(),
    # "fc2_bias_DAGMA_without_h_end": fc2_bias_DAGMA_without_h_end.detach().cpu().tolist(),
    # placeholder for all random restarts
    "random_runs": []
    }

    # print(f'>>> random Init <<<')

    # n_restarts = 5
    # best_random = None
    # best_mle_loss = float("inf")

    # for i in range(n_restarts):
    #     print(f"\n=== Random restart {i+1}/{n_restarts} ===")

    #     eq_model = nonlinear_dce.DagmaMLP_DCE(
    #     dims=[args.d, 10, 1], bias=True)
    #     model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True)
    #     X_random_hat_start = eq_model.forward(X_truth)
    #     mle_loss_random_start = mle_loss(X_random_hat_start, X_truth, torch.eye(args.d))
    #     fc1_weight_random_start = eq_model.fc1.weight
    #     fc1_bias_random_start = eq_model.fc1.bias
    #     fc2_weight_random_start = eq_model.fc2[0].weight
    #     fc2_bias_random_start = eq_model.fc2[0].bias
        
    #     W_est_random, W2_random, x_est_random = model.fit(X_truth, lambda1=3.5e-2, lambda2=5e-3,
    #                                     lr=2e-4, mu_factor=0.1, mu_init=1, T=args.T, warm_iter=7000, max_iter=8000)
    #     fc1_weight_random_end = eq_model.fc1.weight
    #     fc1_bias_random_end = eq_model.fc1.bias
    #     fc2_weight_random_end = eq_model.fc2[0].weight
    #     fc2_bias_random_end = eq_model.fc2[0].bias

    #     _, observed_derivs = eq_model.get_graph(X_truth)
    #     observed_derivs_mean = observed_derivs.mean(dim = 0)
    #     observed_hess_random = eq_model.exact_hessian_diag_avg(X_truth)
    #     Sigma_est_random = eq_model.get_Sigma()
    #     mle_loss_random_end = model.mle_loss(x_est_random, X_truth, Sigma_est_random)
    #     h_val_random = eq_model.h_func(W_est_random, W2_random)
    #     nonlinear_reg_random = eq_model.get_nonlinear_reg(observed_derivs_mean, observed_hess_random)
    #     mle_val = float(mle_loss_random_end .detach().cpu().item())

    #     run_result = {
    #         "run_id": i,
    #         "h_val_random": float(h_val_random.item()),
    #         "mle_loss_random_start": float(mle_loss_random_start.detach().cpu().item()),
    #         "mle_loss_random_end": float(mle_loss_random_end.detach().cpu().item()),
    #         "W_est_random": W_est_random.detach().cpu().tolist(),
    #         "Sigma_est_random": Sigma_est_random.detach().cpu().tolist(),
    #         "nonlinear_reg_random": nonlinear_reg_random.detach().cpu().tolist(),
    #         "fc1_weight_random_start": fc1_weight_random_start.detach().cpu().tolist(),
    #         "fc1_bias_random_start": fc1_bias_random_start.detach().cpu().tolist(),
    #         "fc2_weight_random_start": fc2_weight_random_start.detach().cpu().tolist(),
    #         "fc2_bias_random_start": fc2_bias_random_start.detach().cpu().tolist(),
    #         "fc1_weight_random_end": fc1_weight_random_end.detach().cpu().tolist(),
    #         "fc1_bias_random_end": fc1_bias_random_end.detach().cpu().tolist(),
    #         "fc2_weight_random_end": fc2_weight_random_end.detach().cpu().tolist(),
    #         "fc2_bias_random_end": fc2_bias_random_end.detach().cpu().tolist(),
    #     }
    #     results["random_runs"].append(run_result)

    #     if np.isfinite(mle_val) and mle_val < best_mle_loss:
    #         best_mle_loss = mle_val
    #         best_random = run_result

    # results["best_random_run"] = best_random


    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

    

    



