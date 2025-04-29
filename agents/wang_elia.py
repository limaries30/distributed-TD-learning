import numpy as np
from agents.abstract_distributed_agent import AbstractAgent


class WangElia(AbstractAgent):

    def __init__(self, num_states:int, num_agents:int, num_features:int,
                 gamma : float, agent_config , graph : np.array):
        super().__init__(num_states, num_agents)

        self.num_features = num_features

        self.gamma = gamma
        self.agent_config = agent_config
        self.lr = agent_config["alpha"]


        self.bar_theta = np.random.uniform(-1,1,size=(num_agents*self.num_features,1))  # primary weight
        self.bar_w = np.random.uniform(-1,1,size=(num_agents*self.num_features,1))      # dual weight

        self.identity_matrix =  np.eye(self.num_agents)

        self.laplacian_matrix = graph
        self.bar_laplacian = np.kron(self.laplacian_matrix,np.eye(self.num_features))

    def update(self,current_state:np.array,next_state:np.array,reward:np.array,info):
        '''
            current_state : size of (num_feature,)
            next_state : size of (num_feature,)
            reward : size of (num_agent,)
        
        '''
        next_state= next_state[:,np.newaxis]
        current_state = current_state[:,np.newaxis]
        reward = reward[:,np.newaxis]

        # build large matrices
        A = -current_state@current_state.T+self.gamma*current_state@next_state.T # (num_feature,num_feature)
        barA = np.kron(self.identity_matrix,A)  #(num_agents*num_feature,num_agents*num_feature)
        barR = np.kron(reward,current_state)   #(num_agents*num_feature,1)

        # caculate updates
        primal_delta = barR+barA@self.bar_theta-self.bar_laplacian@self.bar_theta \
                        -self.bar_laplacian@self.bar_w
        
        dual_delta = self.bar_laplacian@self.bar_theta


        self.bar_theta  = self.bar_theta+self.lr*primal_delta
        self.bar_w = self.bar_w + self.lr * dual_delta


  