from agents.abstract_distributed_agent import AbstractAgent
import numpy as np
from utils.math_utils import block_diag_matvec,kron_laplacian_matvec
from utils.graph_utils import get_mixing_matrix
import copy 


class DistributedTD(AbstractAgent):
    '''
    Doan, Thinh, Siva Maguluri, and Justin Romberg. 
    "Finite-time analysis of distributed TD (0) with linear function approximation on multi-agent reinforcement learning."
    '''

    def __init__(self, num_states:int, num_agents:int, num_features:int,
                 gamma : float, agent_config , graph : np.array,
                 mixing_matrix_method : np.array
                 ):
        super().__init__(num_states, num_agents)
        '''
            graph : np.array size of (num_agents,num_agents)
        '''
        self.num_features = num_features

        self.gamma = gamma
        self.agent_config = agent_config
        self.lr = float(agent_config["alpha"])


        self.bar_theta = np.random.uniform(-1,1,size=(num_agents*self.num_features,1))  # primary weight
        self.bar_w = np.zeros((num_agents*self.num_features,1))      # dual weight
        self.prev_semi_grad = np.zeros((num_agents*self.num_features,1))
        self.laplacian_matrix = graph 
        self.mixing_mtarix = get_mixing_matrix(mixing_matrix_method,graph)

        self.steps=0



    def update(self,current_state:np.array,next_state:np.array,reward:np.array,info):
        '''
            current_state : size of (num_feature,)
            next_state : size of (num_feature,)
            reward : size of (num_agent,)
        
        '''
        self.steps+=1

        next_state= next_state[:,np.newaxis]
        current_state = current_state[:,np.newaxis]
        reward = reward[:,np.newaxis]


        A = -current_state@current_state.T+self.gamma*current_state@next_state.T # (num_feature,num_feature)
        barR = np.kron(reward,current_state)   #(num_agents*num_feature,1)
        barAtheta = block_diag_matvec(A,self.bar_theta,self.num_agents)
        yv = kron_laplacian_matvec(self.bar_theta,self.mixing_mtarix,self.num_features)
        self.bar_theta = yv + self.lr * (barR+barAtheta)
