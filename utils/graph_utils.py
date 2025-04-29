import numpy as np
from igraph import Graph
import networkx as nx
import cvxpy as cvx
import copy

def is_connected(g:Graph):
    zero_eig = np.linalg.eig(g)[0][np.isclose(np.linalg.eig(g)[0],0)==True]
    if len(zero_eig)>1:
        return False
    return True


def generate_star_graph(n:int)->np.array:
    '''
        n is number of vertices
        return n by n laplacian graph
    '''
    L = np.diag(np.repeat(1,n))
    #L = np.zeros((n,n))
    L[0,0]=n-1
    L[0,1:]=-1
    L[1:,0]=-1
    
    return L.astype(int)

def generate_ring_graph(n:int)->np.array:
    '''
        n is number of vertices
        return n by n laplacian matrix
    '''
    node_ids = list(range(n)) * 2
    L = np.zeros((n,n))
    for i in range(n):
        if i == n-1:
            L[i,0]= -1
        else:
            L[i,i+1] = -1 
        L[i,i] = 2
        L[i,i-1]=-1
        
    return L.astype(int)


def generate_connected_graph(num_v:int,num_e:int):
    '''
    num_v : nodes, num_e : number of edges
    return n by n laplacian graph
    '''
    for i in range(10):
        g_pre = Graph.Erdos_Renyi(n=num_v, m=num_e)
        g = np.array(g_pre.laplacian(normalized=False))
        if is_connected(g):
            return g
    raise ValueError('The graph is not connected')


def get_graph(graph_type:str,num_agents:int):

    '''
    returns (num_agents,num_agents) graph laplacian matrix 
    '''


    if graph_type=="ring":
        return generate_ring_graph(int(num_agents))
    if graph_type=="random":
        return generate_connected_graph(int(num_agents),int((num_agents-3)*(num_agents-4)/2))
    if graph_type=="star":
        return generate_star_graph(int(num_agents))
    raise ValueError(f'Unknown graph type {graph_type}')


def sinkhorn_iteration(L):
    Y = copy.deepcopy(L)
    n = Y.shape[0]
    arr = np.ones((n,n))#np.random.rand(n,n)
    ids  = Y==0
    arr[ids]=0
    
    for it in range(100):
        D1 = np.diagflat(1. /np.sum(arr,axis=1))
        D2 = np.diagflat(1. /np.sum(np.dot(D1,arr),axis=0))
        arr = np.dot(np.dot(D1,arr),D2)
    return arr


def generate_mixing_matrix(graph:np.array):
    '''
        From https://github.com/liboyue/Network-Distributed-Algorithm/blob/master/nda/optimizers/utils.py
    '''

    n = graph.shape[0]#10
    ind = - graph*(graph==-1) + np.eye(n)

    ind = ~ind.astype(bool)

    average_matrix = np.ones((n, n)) / n
    one_vec = np.ones(n)

    W = cvx.Variable((n, n))

    if ind.sum() == 0:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W == W.T,
                                cvx.sum(W, axis=1) == one_vec,
                                cvx.diag(W)>0,
                                W >= 0
                            ])
    else:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W[ind] == 0,
                                W == W.T,
                                cvx.diag(W)>=0.00001,
                                W >=0,
                                cvx.sum(W, axis=1) == one_vec
                            ])
    prob.solve()

    W = W.value
    W[ind] = 0
    W -= np.diag(W.sum(axis=1) - 1)

    if not (np.diag(W)>0).all():
        print(W)
        raise ValueError

    return W