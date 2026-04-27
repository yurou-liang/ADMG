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
import time
import bnlearn as bn
from itertools import combinations
from causallearn.utils.Dataset import load_dataset

@torch.no_grad()
def init_union_uniform_(tensor, low=0.5, high=2.0, generator=None):
    mags = torch.empty_like(tensor).uniform_(low, high, generator=generator)
    signs = torch.empty_like(tensor).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
    tensor.copy_(mags * signs)

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
        rng = np.random.RandomState(seed)

    # --- Step 1: generate a DAG (acyclic directed structure)
    A_dir = np.triu((rng.random((d, d)) < p_dir).astype(int), 1) # generate random 0/1 matrix with edge prob p_dir, keep only upper triangular part for acyclicty
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
            if rng.random() < p_bidir:
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

def generate_bowfree_admg(d, p_dir=0.4, p_bidir=0.3, seed=None):
    """
    Generate a bow-free ancestral ADMG:
        {node: {'parents': [], 'spouses': []}}
    where spouses = bidirected connections (↔).

    Bow-free means:
        if i -> j exists, then i <-> j cannot exist.

    Ancestral means:
        if i <-> j exists, then i is not an ancestor of j
        and j is not an ancestor of i.

    Parameters
    ----------
    d : int
        Number of nodes.
    p_dir : float
        Probability of a directed edge.
    p_bidir : float
        Probability of a bidirected edge.
    seed : int or None
        Random seed.

    Returns
    -------
    admg : dict
        Graph representation:
        admg[j] = {"parents": [...], "spouses": [...]}
    A_dir : np.ndarray
        Directed adjacency matrix, where A_dir[i, j] = 1 means i -> j.
    A_bidir : np.ndarray
        Symmetric bidirected adjacency matrix,
        where A_bidir[i, j] = 1 means i <-> j.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # Step 1: generate a DAG (acyclic directed graph)
    A_dir = np.triu((rng.random((d, d)) < p_dir).astype(int), 1)

    # Step 2: generate bidirected edges, excluding bows
    A_bidir = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p_bidir:
                # Bow-free condition:
                # do not add i <-> j if i -> j or j -> i exists
                if A_dir[i, j] == 0 and A_dir[j, i] == 0:
                    A_bidir[i, j] = 1
                    A_bidir[j, i] = 1

    # Step 3: return graph as dict
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
        g = torch.Generator()
        g.manual_seed(seed)
    bias=True
    fc1 = nn.Linear(d, d * dims[1], bias=bias) # [d * dims[1], d]
    init_union_uniform_(fc1.weight, generator=g)
    if fc1.bias is not None:
        init_union_uniform_(fc1.bias, generator=g)
    # self.fc1.weight.bounds = self._bounds()
    mask = torch.ones(d * dims[1], d)

    for j in range(d):
        allowed_parents = admg[j]['parents']
        not_parents = [p for p in range(d) if p not in allowed_parents]
        mask[j * dims[1]:(j + 1) * dims[1], not_parents] = 0.0 + 1e-6

    layers = []
    for l in range(len(dims) - 2):
        layer = LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias)

        init_union_uniform_(layer.weight, generator=g)
        if layer.bias is not None:
            init_union_uniform_(layer.bias, generator=g)
        layers.append(layer)
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
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    d = A_bidir.shape[0]

    # Step 1: random strong correlations for existing edges
    R = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            if A_bidir[i, j]:
                val = rng.uniform(low, high)
                # Randomly flip sign for variety (optional)
                if rng.random() < 0.5:
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

def generate_from_epsilon(dims, epsilon, fc1, fc2, mask, parents, order):
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

def f_exp(x):
    return torch.exp(-x**2)

def f_tanh(x):
    return torch.tanh(x)

def f_sin(x):
    return torch.sin(x)

FUNCTIONS = {
    "exp": f_exp,
    "tanh": f_tanh,
    "sin": f_sin,
}

def sample_edge_mechanisms(parents):
    """
    For each directed edge k -> i, sample:
      - a nonlinear function from {exp(-x^2), tanh(x), sin(x)}
      - a weight from [-1.5, -0.5] U [0.5, 1.5]

    Returns:
      edge_func_name: dict {(k, i): function_name}
      edge_weight: dict {(k, i): weight}
    """
    # rng = np.random.RandomState(seed)

    func_names = ["exp", "tanh", "sin"]

    edge_func_name = {}
    edge_weight = {}

    for i, pa_i in parents.items():
        for k in pa_i:
            edge_func_name[(k, i)] = np.random.choice(func_names)

            # sample sign first, then magnitude
            sign = np.random.choice([-1.0, 1.0])
            magnitude = np.random.uniform(2, 3)
            edge_weight[(k, i)] = float(sign * magnitude)

    return edge_func_name, edge_weight

def generate_from_func(epsilon, parents, order, edge_func_name, edge_weight):
    """
    Generate data from:
        X_i = sum_{k in pa(i)} w_{k,i} f_{k,i}(X_k) + epsilon_i

    Args:
        epsilon: [n, d] tensor
        parents: dict {i: [parents_of_i]}
        order: topological order of nodes
        edge_func_name: dict {(k, i): "exp"|"tanh"|"sin"}
        edge_weight: dict {(k, i): float}

    Returns:
        X: [n, d] tensor
    """
    n, d = epsilon.shape
    X = torch.zeros_like(epsilon)

    for i in order:
        if len(parents[i]) == 0:
            X[:, i] = epsilon[:, i]
        else:
            total = torch.zeros(n, dtype=epsilon.dtype)

            for k in parents[i]:
                fname = edge_func_name[(k, i)]
                w = edge_weight[(k, i)]
                f = FUNCTIONS[fname]

                total = total + w * f(X[:, k])

            X[:, i] = total + epsilon[:, i]

    return X

def g_single(x, parents, order, edge_func_name, edge_weight):
    """
    x: [d]
    returns g(x): [d], where
        g_i(x) = sum_{k in pa(i)} w_{k,i} f_{k,i}(x_k)
    """
    d = x.shape[0]
    out = torch.zeros_like(x)

    for i in order:
        total = 0.0
        for k in parents[i]:
            fname = edge_func_name[(k, i)]
            w = edge_weight[(k, i)]
            f = FUNCTIONS[fname]
            total = total + w * f(x[k])
        out[i] = total

    return out

def build_confounder_map_from_admg(admg):
    """
    Each bidirected edge i <-> j gets one latent confounder.

    Returns
    -------
    spouse_edge_to_conf : dict
        {(i,j) with i<j: confounder_id}
    """
    d = len(admg)
    spouse_edge_to_conf = {}

    conf_id = 0
    for i in range(d):
        for j in admg[i]["spouses"]:
            a, b = min(i, j), max(i, j)
            if (a, b) not in spouse_edge_to_conf:
                spouse_edge_to_conf[(a, b)] = conf_id
                conf_id += 1

    return spouse_edge_to_conf

def sample_latent_confounders(n, m, seed=None, dtype=torch.float32, device=None):
    std = 1
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        U = std * torch.randn(n, m, generator=g, dtype=dtype, device=device)
    else:
        U = std * torch.randn(n, m, dtype=dtype, device=device)
    return U

def generate_confounder_layers_from_admg(d, dims_u, admg, seed=None):
    """
    Confounder network g_j(u_sp(j)) using the same input/output layout as generate_layers.

    dims_u should be [d, hidden_dim, 1]
    """

    bias = True
    fc1_u = nn.Linear(d, d * dims_u[1], bias=bias)
    init_union_uniform_(fc1_u.weight, low=1.5, high=2.0)
    if fc1_u.bias is not None:
        init_union_uniform_(fc1_u.bias, low=1.5, high=2.0)

    mask_u = torch.ones(d * dims_u[1], d)

    # node j is only allowed to use spouse-columns
    for j in range(d):
        allowed_spouses = admg[j]["spouses"]
        not_spouses = [k for k in range(d) if k not in allowed_spouses]
        mask_u[j * dims_u[1]:(j + 1) * dims_u[1], not_spouses] = 0.0

    layers = []
    for l in range(len(dims_u) - 2):
        layer = LocallyConnected(d, dims_u[l + 1], dims_u[l + 2], bias=bias)
        init_union_uniform_(layer.weight, low=1.5, high=2.0)
        if layer.bias is not None:
            init_union_uniform_(layer.bias, low=1.5, high=2.0)
        layers.append(layer)

    fc2_u = nn.ModuleList(layers)
    return fc1_u, fc2_u, mask_u

def generate_from_epsilon_additive_confounders(
    dims_x, epsilon, fc1_x, fc2_x, mask_x,
    dims_u, U, spouse_edge_to_conf, fc1_u, fc2_u, mask_u,
    admg, parents, order
):
    """
    Generate:
        X_j = f_j(X_pa(j)) + g_j(U_sp(j)) + epsilon_j

    where each bidirected edge j <-> k has one shared latent confounder u_{jk}.
    """
    n, d = epsilon.shape
    X = torch.zeros_like(epsilon)

    for j in order:
        # causal part f_j(X_pa(j))
        if len(parents[j]) == 0:
            fx_j = torch.zeros(n, dtype=epsilon.dtype, device=epsilon.device)
        else:
            x_partial = torch.zeros(n, d, dtype=epsilon.dtype, device=epsilon.device)
            for p in parents[j]:
                x_partial[:, p] = X[:, p]

            f_out = forward(dims_x, fc1_x, fc2_x, mask_x, x_partial)  # [n, d]
            fx_j = f_out[:, j]

        # confounder part g_j(U_sp(j))
        spouses_j = admg[j]["spouses"]
        if len(spouses_j) == 0:
            fu_j = torch.zeros(n, dtype=epsilon.dtype, device=epsilon.device)
        else:
            u_partial = torch.zeros(n, d, dtype=epsilon.dtype, device=epsilon.device)

            for k in spouses_j:
                a, b = min(j, k), max(j, k)
                conf_id = spouse_edge_to_conf[(a, b)]
                u_partial[:, k] = U[:, conf_id]

            g_out = forward(dims_u, fc1_u, fc2_u, mask_u, u_partial)  # [n, d]
            fu_j = g_out[:, j]

        X[:, j] = fx_j + fu_j + epsilon[:, j]

    return X

def f_x_single(x_single):
    return forward(dims, fc1_x, fc2_x, mask_x, x_single.unsqueeze(0)).squeeze(0)

def sample_cov(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)

def compute_f_of_X(dims_x, fc1_x, fc2_x, mask_x, X, parents):
    n, d = X.shape
    F = torch.zeros_like(X)

    for j in range(d):
        if len(parents[j]) == 0:
            continue

        x_partial = torch.zeros_like(X)
        for p in parents[j]:
            x_partial[:, p] = X[:, p]

        f_out = forward(dims_x, fc1_x, fc2_x, mask_x, x_partial)
        F[:, j] = f_out[:, j]

    return F

def mle_loss(output: torch.Tensor, target: torch.Tensor, Sigma: torch.Tensor):
        """Computes the MLE loss 1/n*Tr((X-X_est)Sigma^{-1}(X-X_est)^T)"""
        n, d = target.shape
        tmp = torch.linalg.solve(Sigma, (target - output).T)
        mle = torch.trace((target - output)@tmp)/n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle

def dag_to_admg_with_hidden_common_cause(dag_parents, hidden_node, observed_nodes):
    from itertools import combinations

    children = {v: [] for v in dag_parents}
    for child, parents in dag_parents.items():
        for p in parents:
            children[p].append(child)

    node_to_idx = {v: i for i, v in enumerate(observed_nodes)}

    admg = {
        node_to_idx[v]: {"parents": [], "spouses": []}
        for v in observed_nodes
    }

    # directed edges among observed nodes
    for child, parents in dag_parents.items():
        if child not in node_to_idx:
            continue

        child_idx = node_to_idx[child]
        for p in parents:
            if p in node_to_idx:
                admg[child_idx]["parents"].append(node_to_idx[p])

    # bidirected clique among children of hidden_node
    hidden_children = [c for c in children[hidden_node] if c in node_to_idx]

    for u, v in combinations(hidden_children, 2):
        i, j = node_to_idx[u], node_to_idx[v]
        admg[i]["spouses"].append(j)
        admg[j]["spouses"].append(i)

    for i in admg:
        admg[i]["parents"] = sorted(set(admg[i]["parents"]))
        admg[i]["spouses"] = sorted(set(admg[i]["spouses"]))

    return admg, node_to_idx



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Comparison between RKHADagma and NOTEARS',)

    parser.add_argument('-realData', default="n", type=str)
    parser.add_argument('-d', '--num_nodes', dest='d', default=4, type=int)
    parser.add_argument('-g', '--graph_type', dest='g', default="ancestral", type=str)
    parser.add_argument('-s', '--seed', dest='s',  default=42, type=int)
    parser.add_argument('-a', '--admg', dest='a',  default=3, type=int)
    parser.add_argument('-f', '--function', dest='f', default="MLP", type=str)
    parser.add_argument('-T', '--num_iterations', dest='T', default=5, type=int)
    parser.add_argument('-lambda1', default=0.001, type=float)
    parser.add_argument('-lambda_corr', default=0.1, type=float)
    parser.add_argument('-lambda_nl', default=8.0, type=float)

    args = parser.parse_args()
    torch.set_default_dtype(torch.double)
    np.random.seed(args.s)
    torch.manual_seed(args.s)

    if args.realData == "y":
        data, labels = load_dataset("sachs")

        # make column names match bnlearn convention
        rename = {
            "pka": "PKA",
            "pkc": "PKC",
            "p38": "P38",
            "pip2": "PIP2",
            "pip3": "PIP3",
            "plc": "Plcg",
            "raf": "Raf",
            "mek": "Mek",
            "erk": "Erk",
            "akt": "Akt",
            "jnk": "Jnk",
        }

        labels = [rename.get(str(x).lower(), str(x)) for x in labels]
        df = pd.DataFrame(data, columns=labels)

        # ground-truth DAG from bnlearn
        ground_truth_dag = bn.import_DAG("sachs", verbose=False)
        G = ground_truth_dag["model"]

        # reorder data columns according to bnlearn graph nodes
        graph_nodes = list(G.nodes())
        df = df[graph_nodes]
        hidden_var = "PKC"

        df_obs = df.drop(columns=[hidden_var])
        df_obs = df_obs.sample(n=2000, random_state=0).reset_index(drop=True)
        X_truth = torch.tensor(df_obs.values, dtype=torch.float64)
        args.d = X_truth.shape[1]
        var_names = list(df_obs.columns)
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        dag_parents = {node: [] for node in graph_nodes}
        for parent, child in G.edges():
            dag_parents[child].append(parent)
        dag_parents = {k: sorted(v) for k, v in dag_parents.items()}
        print(dag_parents)
        var_names = list(df_obs.columns)

        admg, node_to_idx = dag_to_admg_with_hidden_common_cause(
            dag_parents=dag_parents,
            hidden_node="PKC",
            observed_nodes=var_names
        )

        filename = f'realData.json'
        results = {
        'admg': admg,
        'lambda1': args.lambda1,
        'lambda_corr': args.lambda_corr,
        'lambda_nl': args.lambda_nl,
        'X_truth': X_truth.detach().cpu().numpy().tolist(),
        "random_runs": []
        }
    
    else:

        if args.g == "ancestral":
            print(f'>>> Generating Ancestral ADMG <<<')
            admg, A_dir, A_bidir = generate_ancestral_admg(args.d, p_dir=1/(args.d-1), p_bidir=1/(args.d-1), seed=args.a)
        elif args.g == "bowfree":
            print(f'>>> Generating Bow-free ADMG <<<')
            admg, A_dir, A_bidir = generate_bowfree_admg(args.d, p_dir=1/(args.d-1), p_bidir=1/(args.d-1), seed=args.a)
        print("admg: ", admg)
        parents = {j: admg[j]['parents'] for j in admg}
        G = nx.DiGraph()
        G.add_nodes_from(parents.keys())
        for j, pa in parents.items():
            G.add_edges_from((p, j) for p in pa)
        order = list(nx.topological_sort(G))

        n_samples = 2000 

        if args.f == "MLP":
            print(f'>>> Generating Data with MLP <<<')
            Sigma_truth = generate_covariance(A_bidir, seed=args.s)
            epsilon = np.random.multivariate_normal([0] * args.d, Sigma_truth, size=n_samples)
            Sigma_truth = torch.tensor(Sigma_truth)
            epsilon = torch.tensor(epsilon)
            dims=[args.d, 100, 1]
            fc1, fc2, mask = generate_layers(args.d, dims, admg, seed = args.s)
            # scale_weights(fc1, factor=10)
            # for layer in fc2:
            #     scale_weights(layer, factor=15)
            X_truth = generate_from_epsilon(dims, epsilon, fc1, fc2, mask, parents, order).detach()
            J = vmap(jacrev(f))(X_truth)    # shape [n_samples, d, d]
            X = X_truth - epsilon
        elif args.f == "func":
            print(f'>>> Generating Data with Explicit Functions <<<')
            Sigma_truth = generate_covariance(A_bidir, seed=args.s)
            epsilon = np.random.multivariate_normal([0] * args.d, Sigma_truth, size=n_samples)
            Sigma_truth = torch.tensor(Sigma_truth)
            epsilon = torch.tensor(epsilon)
            edge_func_name, edge_weight = sample_edge_mechanisms(parents)
            X_truth = generate_from_func(epsilon, parents, order, edge_func_name, edge_weight).detach()
            J = vmap(jacrev(lambda x: g_single(x, parents, order, edge_func_name, edge_weight)))(X_truth)    # shape [n_samples, d, d]
            X = X_truth - epsilon
            
        elif args.f == "nonlinear_confounder":
            print(f'>>> Generating Data with Nonlinear Confounders <<<')
            dims=[args.d, 100, 1]
            dims_u = [args.d, 5, 1]
            epsilon = 0.7*np.random.multivariate_normal([0] * args.d, np.eye(args.d), size=n_samples)
            epsilon = torch.tensor(epsilon)
            fc1_x, fc2_x, mask_x = generate_layers(args.d, dims, admg, seed=args.s)

            # spouse-edge confounder structure
            spouse_edge_to_conf = build_confounder_map_from_admg(admg)
            m = len(spouse_edge_to_conf)

            # latent confounders, one per spouse edge
            U = sample_latent_confounders(
                n=n_samples,
                m=m,
                seed=13,
                dtype=epsilon.dtype,
                device=epsilon.device
            )

            # confounder network, same input/output format as observed network
            fc1_u, fc2_u, mask_u = generate_confounder_layers_from_admg(
                d=args.d,
                dims_u=dims_u,
                admg=admg,
            )

            X_truth = generate_from_epsilon_additive_confounders(
                dims_x=dims,
                epsilon=epsilon,
                fc1_x=fc1_x,
                fc2_x=fc2_x,
                mask_x=mask_x,
                dims_u=dims_u,
                U=U,
                spouse_edge_to_conf=spouse_edge_to_conf,
                fc1_u=fc1_u,
                fc2_u=fc2_u,
                mask_u=mask_u,
                admg=admg,
                parents=parents,
                order=order,
            ).detach()

            J = vmap(jacrev(f_x_single))(X_truth) 
            F_x = compute_f_of_X(dims, fc1_x, fc2_x, mask_x, X_truth, parents)
            X = F_x
            R = X_truth - F_x   # this is g(U) + epsilon
            Sigma_truth = sample_cov(R)
        else:
            raise ValueError(f"Unsupported function type: {args.f}")
        
        W_truth = torch.sqrt(torch.mean(J ** 2, axis=0).T)
        print("W_truth: ", W_truth)
        
        mle_loss_truth = mle_loss(X, X_truth, Sigma_truth)
        M_truth = reverse_SPDLogCholesky(Sigma_truth)

        filename = f'result_d{args.d}_graph{args.g}_admg{args.a}_seed{args.s}_f{args.f}.json'
        results = {
        'admg': admg,
        'lambda1': args.lambda1,
        'lambda_corr': args.lambda_corr,
        'lambda_nl': args.lambda_nl,
        'X_truth': X_truth.detach().cpu().numpy().tolist(),
        'X': X.detach().cpu().numpy().tolist(),
        'epsilon': epsilon.detach().cpu().numpy().tolist(),
        'W_truth': W_truth.detach().cpu().numpy().tolist(),
        # 'W_start': W_truth_start.detach().cpu().numpy().tolist(),
        "Sigma_truth": Sigma_truth.tolist(),
        "M_truth": M_truth.detach().cpu().tolist(),
        # 'h_val_truth': h_val_truth.item(),
        'mle_loss_truth': mle_loss_truth.detach().cpu().tolist(),
        "random_runs": []
        }


    print(f'>>> random Init <<<')

    n_restarts = 1
    best_random = None
    best_mle_loss = float("inf")

    for i in range(n_restarts):
        print(f"\n=== Random restart {i+1}/{n_restarts} ===")

        # eq_model = nonlinear.DagmaMLP(
        # dims=[args.d, 10, 1], bias=True, dtype=torch.double)
        # model = nonlinear.DagmaNonlinear(
        #     eq_model, dtype=torch.double, use_mse_loss=True)

        # W_est_dagma_with_h, X_est = model.fit(X_truth, lambda1=2e-2, lambda2=0.005,
        #                         T=5, lr=2e-4, w_threshold=0.3, mu_init=1, warm_iter=70000, max_iter=80000, consider_h=True)

        
        # E = X_truth - X_est
        # E_centered = E - E.mean(dim=0, keepdim=True)
        # var_mle = (E_centered ** 2).mean(dim=0)
        # var_max = var_mle.max()
        # var_mle_const = var_max.repeat(var_mle.shape[0])

        # ✅ detach + clone for safety (important for deepcopy)
        # var_mle_const = var_mle_const.detach().clone()

        run_start_time = time.perf_counter()
        eq_model = nonlinear.DagmaMLP(
        dims=[args.d, 10, 1], bias=True, dtype=torch.double)
        model = nonlinear.DagmaNonlinear(
            eq_model, dtype=torch.double, use_mse_loss=True)

        W_est_dagma_random = model.fit(X_truth, lambda1=2e-2, lambda2=0.005,
                                T=1, lr=2e-4, w_threshold=0.3, mu_init=1, warm_iter=70000, max_iter=80000, consider_h=False)

        # Use DAGMA weights as initial weights for DAGMA-DCE
        fc1_weight_random = eq_model.fc1.weight
        fc1_bias_random = eq_model.fc1.bias
        fc2_weight_random = eq_model.fc2[0].weight
        fc2_bias_random= eq_model.fc2[0].bias

        # eq_model = nonlinear_dce.DagmaMLP_DCE(
        #     dims=[args.d, 10, 1], diag0= var_mle_const, bias=True)
        eq_model = nonlinear_dce.DagmaMLP_DCE(
            dims=[args.d, 10, 1], bias=True)
        model = nonlinear_dce.DagmaDCE(eq_model, use_mle_loss=True, graph_type=args.g)
        eq_model.fc1.weight = fc1_weight_random
        eq_model.fc1.bias = fc1_bias_random
        eq_model.fc2[0].weight = fc2_weight_random
        eq_model.fc2[0].bias = fc2_bias_random
        X_random_hat_start = eq_model.forward(X_truth)
        mle_loss_random_start = mle_loss(X_random_hat_start, X_truth, torch.eye(args.d))

        W_est_random, W2_random, x_est_random = model.fit(X_truth, lambda1=args.lambda1, lambda_corr=args.lambda_corr, lambda_nl=args.lambda_nl, lambda2=5e-3,
                                        lr=2e-4, mu_factor=0.1, mu_init=1, T=5, warm_iter=7000, max_iter=8000)
        run_time_sec = time.perf_counter() - run_start_time

            
        fc1_weight_random_end = eq_model.fc1.weight
        fc1_bias_random_end = eq_model.fc1.bias
        fc2_weight_random_end = eq_model.fc2[0].weight
        fc2_bias_random_end = eq_model.fc2[0].bias

            
        _, observed_derivs = eq_model.get_graph(X_truth)
        observed_derivs_mean_random = observed_derivs.mean(dim = 0)
        observed_hess_random = eq_model.exact_hessian_diag_avg(X_truth)
        Sigma_est_random = eq_model.get_Sigma()
        mle_loss_random_end = model.mle_loss(x_est_random, X_truth, Sigma_est_random)
        h_val_random = eq_model.h_func(W_est_random, W2_random)
        nonlinear_reg_random = eq_model.get_nonlinear_reg(observed_derivs_mean_random, observed_hess_random)
        mle_val = float(mle_loss_random_end .detach().cpu().item())

        run_result = {
            "run_id": i,
            "h_val_random": float(h_val_random.item()),
            # "var_mle_const": var_mle_const.detach().cpu().tolist(),
            "mle_loss_random_start": float(mle_loss_random_start.detach().cpu().item()),
            "mle_loss_random_end": float(mle_loss_random_end.detach().cpu().item()),
            "W_est_random": W_est_random.detach().cpu().tolist(),
            "Sigma_est_random": Sigma_est_random.detach().cpu().tolist(),
            "x_est_random": x_est_random.detach().cpu().tolist(),
            "run_time_sec": run_time_sec,
            "nonlinear_reg_random": nonlinear_reg_random.detach().cpu().tolist(),
            "fc1_weight_random_start": fc1_weight_random.detach().cpu().tolist(),
            "fc1_bias_random_start": fc1_bias_random.detach().cpu().tolist(),
            "fc2_weight_random_start": fc2_weight_random.detach().cpu().tolist(),
            "fc2_bias_random_start": fc2_bias_random.detach().cpu().tolist(),
            "fc1_weight_random_end": fc1_weight_random_end.detach().cpu().tolist(),
            "fc1_bias_random_end": fc1_bias_random_end.detach().cpu().tolist(),
            "fc2_weight_random_end": fc2_weight_random_end.detach().cpu().tolist(),
            "fc2_bias_random_end": fc2_bias_random_end.detach().cpu().tolist(),
        }
        results["random_runs"].append(run_result)

        # if np.isfinite(mle_val) and mle_val < best_mle_loss:
        #     best_mle_loss = mle_val
        #     best_random = run_result

        # results["best_random_run"] = best_random


        with open(filename, 'w') as file:
            json.dump(results, file, indent=4) 

    

    



