import numpy as np


def kron_laplacian_matvec(w,graph,d):
    N = graph.shape[0]
    w_reshaped = w.reshape(N,d)
    result = graph @ w_reshaped
    return result.reshape(N*d,1)

def block_diag_matvec(A,vec,num_agents):
    d = A.shape[0]
    return np.vstack([A@vec[i*d:(i+1)*d] for i in range(num_agents)])