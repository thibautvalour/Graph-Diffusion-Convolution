import torch

def compute_Tsym(adj): 
    ''' adj is a sparse matrix'''
    N = adj.shape[0]
    D = torch.sparse.sum(adj, dim=1).to_dense() # get degree matrix D
    D_sqrt_inv = torch.pow(D, -0.5)
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1)
    D_sqrt_inv = torch.sparse_coo_tensor(indices, D_sqrt_inv,
                                         size=(N, N))
    
    Tsym = D_sqrt_inv.matmul(adj).matmul(D_sqrt_inv)
    return Tsym
