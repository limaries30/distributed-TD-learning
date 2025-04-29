import numpy as np



def get_mdp(env_name:str,args,graph):
    from mdps.distributed_random_mdp import DistributedRandomMDP
    from mdps.distributed_mdp_1 import DistributedMDP_1

    if env_name=="DistributedRandomMDP":
        return DistributedRandomMDP(num_states=args.num_states,num_features=args.num_features,
                                    gamma=args.gamma,
                                    num_agents=args.num_agents,feature_name=args.feature_name,
                                    graph = graph
                                    )
    if env_name=="DistributedMDP_1":
        return DistributedMDP_1(num_states=args.num_states,num_features=args.num_features,
                                    gamma=args.gamma,
                                    num_agents=args.num_agents,feature_name=args.feature_name,
                                    graph = graph)
    raise ValueError(f"Unknown env name {env_name}")


def get_stationary_distribution(transition_matrix:np.ndarray):
    '''
    
    input : transition matrix of Markov chain ( shape: =(|S||A|, |S||A|) )
    returns stationary distribution of Markov chain  (shape:(|S||A|,))
    
    '''
    
    transition_matrix_transp = transition_matrix.T
    eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp) # To compute the eigenvalues and eigenvectors of the transition matrix
    
    close_to_1_idx = np.isclose(eigenvals,1)
    target_eigenvect = eigenvects[:,close_to_1_idx]
    target_eigenvect = target_eigenvect[:,0]                        # Turn the eigenvector elements into probabilites
    
    stationary_distrib = target_eigenvect / sum(target_eigenvect)
    return np.abs(stationary_distrib)




def make_prob_vector(p:np.ndarray):
    '''
        input p: a vector of shape (m,)
        returns a probability vector of shape (m,)
    '''

    p = np.exp(p)
    p = p/np.sum(p)
    p_tmp = np.floor(100*p)/100             # round to 2 decimal points
    p_tmp[-1] = 1-np.sum(p_tmp[:-1])
    return p_tmp

def make_prob_matrix(n:int,m:int,limit=2):
    '''
        return a row stochastic matrix of shape (n,m)
        UNIFORMLY sampled from [-limit,limit] and then normalized to probability vector    
    '''
    p = np.array([make_prob_vector(np.random.uniform(-limit,limit,m)) for i in range(n)])
    return p


def make_random_behavior_policy_matrix(num_states:int,num_actions:int):
    '''
    return a behavior policy matrix of shape (|S|, |S||A|)
    '''
    Pib = np.zeros((num_states,num_states*num_actions))
    beta = make_prob_matrix(num_states,num_actions)
    
    for s in range(num_states):
        es = np.zeros((num_states,1)).T
        es[0,s]=1
        Pib[s,]= np.kron(es,beta[s,:])
    
    return Pib

