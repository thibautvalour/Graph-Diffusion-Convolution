import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_Tsym(adj): 
    ''' adj is a sparse matrix'''
    N = adj.shape[0]
    D = torch.sparse.sum(adj, dim=1).to_dense() # get degree matrix D
    D_sqrt_inv = torch.pow(D, -0.5)
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1).to(device)
    D_sqrt_inv = torch.sparse_coo_tensor(indices, D_sqrt_inv,
                                         size=(N, N))
    
    Tsym = D_sqrt_inv.matmul(adj).matmul(D_sqrt_inv)
    return Tsym

def gdc_pagerank(A, alpha, eps):

    N = A.shape[0]

    # Self-loops
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1).to(device) 
    values = torch.ones(N, dtype=torch.float).to(device) 
    sparse_identiy = torch.sparse_coo_tensor(indices, values,
                                             size=(N, N))    
    
    A_loop = A + sparse_identiy
    
    # Symmetric transition matrix
    D_loop = torch.sparse.sum(A_loop, dim=1).to_dense()
    D_sqrt_inv = torch.pow(D_loop, -0.5)
    D_sqrt_inv = torch.sparse_coo_tensor(indices, D_sqrt_inv,
                                         size=(N, N))


    T_sym = D_sqrt_inv @ A_loop @ D_sqrt_inv

    # PPR-based diffusion
    S = alpha * torch.pow(sparse_identiy-(1-alpha)*T_sym, -1)

    # TODO : check why negative values are present in S
    # Sparsify using threshold epsilon
    indices = S.indices()
    thresholded_val = S.values() * (S.values() >= eps)
    S_tilde = torch.sparse_coo_tensor(indices, thresholded_val,
                                      size=(N, N))

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = torch.sparse.sum(S_tilde, dim=1).to_dense()
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1).to(device) 
    D_tilde_vec = torch.sparse_coo_tensor(indices, D_tilde_vec,
                                            size=(N, N))
    T_S = S_tilde @ torch.pow(D_tilde_vec, -1)
    
    return T_S
