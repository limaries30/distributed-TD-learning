from agents.abstract_distributed_agent import AbstractAgent
import numpy as np
from utils.math_utils import block_diag_matvec,kron_laplacian_matvec
from utils.graph_utils import get_mixing_matrix
import copy 


class GradientTracking(AbstractAgent):
    '''
     Decentralized TD Tracking with Linear Function Approximation and its Finite-Time Analysis
     https://proceedings.neurips.cc/paper/2020/file/9ec51f6eb240fb631a35864e13737bca-Paper.pdf
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



        barWtheta = kron_laplacian_matvec(self.bar_theta,self.mixing_mtarix,self.num_features)
        self.bar_theta  = barWtheta + self.lr * self.bar_w 
        barAtheta = block_diag_matvec(A,self.bar_theta,self.num_agents)
        
        curent_semi_grad =  barR+barAtheta
        barLw =   kron_laplacian_matvec(self.bar_w,self.mixing_mtarix,self.num_features)
        self.bar_w = barLw + curent_semi_grad -self.prev_semi_grad
        self.prev_semi_grad = copy.deepcopy(curent_semi_grad)
