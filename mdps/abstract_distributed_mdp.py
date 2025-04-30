import numpy as np
from abc import ABC,abstractmethod
from utils.mdp_utils import get_stationary_distribution
import copy


class AbstractDistributedMDP(ABC):

    def __init__(
        self,num_states:int,
        num_features:int,
        num_agents:int,
        X:np.ndarray,
        P:np.ndarray,
        barR:np.ndarray,
        gamma:float,
        ):

        '''
         X:  (|S|, num_features)
         P: |S| \times |S| (transition matrix)
         R : (N,|S|,|S|)  (reward vector)
         gamma: discount factor
        '''

        print('bar',barR.shape)
        
        self.num_states = num_states
        self.num_features =  num_features
        self.num_agents = num_agents


        # Validate shapes (optional, can be removed for performance)
        assert X.shape == (num_states , num_features), "X has incorrect shape"
        assert P.shape == (num_states, num_states), "P has incorrect number of rows"
        assert barR.shape == (num_agents,num_states,num_states ), "R has incorrect shape"
        assert 0 <= gamma <= 1, "gamma should be in [0, 1]"


        self.X =X
        self.P = P
        self.barR = barR
        self.dbeta = get_stationary_distribution(self.P)
        self.Dbeta = np.diag(self.dbeta)
        self.gamma = gamma


        self.current_state_idx = None


    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def step(self):
        pass

