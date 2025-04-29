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


    def compute_pbe_error(self,theta:np.ndarray):
        '''
            Compute the error of the policy evaluation
        '''

        X = self.X
        P = self.P
        Dbeta = self.Dbeta
        R = self.R
        gamma = self.gamma
        
        Proj = X@np.linalg.pinv(X.T@Dbeta@X)@X.T@Dbeta

        
        Pi = construct_greedy_matrix_Q(self.num_states,self.num_actions,X@theta)

        PBE_M = Proj@(R+gamma*P@Pi@X@theta) - X@theta

        PBE_error = PBE_M.T@Dbeta@PBE_M 
        return PBE_error[0,0]

    def print_expected_sum_following_pi(self,Pi:np.ndarray):
        '''
            Print the expected sum of rewards following policy Pi
        '''
        tmp = 0    
        P = np.eye(self.num_states*self.num_actions)
        for i in range(500):
            
            tmp+=(self.gamma**i)*P@self.R
            P = self.P@Pi@P
        print('Expected sum of rewards following policy Pi:',tmp)


    def find_optimal_policy(self):

        prev_tmp = -np.inf 
        Pis = construct_greedy_matrices(num_states=self.num_states,num_actions=self.num_actions)
        prev_opt = label_policy(self.num_states,self.num_actions,Pis[0])
        for Pi in Pis:
            tmp  = 0    
            P = np.eye(self.num_states*self.num_actions)
            for i in range(500):
                
                tmp+=(self.gamma)**i*P@self.R
                P = self.P@Pi@P
            if (tmp>=prev_tmp).all():
                prev_tmp = copy.deepcopy(tmp)
                prev_opt = label_policy(self.num_states,self.num_actions,Pi)
        return prev_opt
    

    def check_possible_sols(self):


        X = self.X
        Dbeta = self.Dbeta
        P = self.P
        gamma = self.gamma
        R=  self.R
        gamma = self.gamma
        I = np.eye(self.num_features)
        Pis = construct_greedy_matrices(self.num_states,self.num_actions)

        sol_results = []
        sols = []
        is_singulars = []

        for Pi in Pis:
            A = X.T@Dbeta@X-gamma*X.T@Dbeta@P@Pi@X
        
            if np.linalg.matrix_rank(A)<self.num_features: # singular A
                sol_results.append(False)
                is_singulars.append(True)
                sols.append(None)
                continue

            sol_candidate = np.linalg.pinv(A)@X.T@Dbeta@R
            Qstar = np.repeat(Pi@(X@sol_candidate),self.num_actions,axis=0)


            is_not_sol = (~(Qstar>=(X@sol_candidate-(1e-6)))).any()

            if is_not_sol:
                sol_results.append(False)
                sols.append(sol_candidate)
            else:
                sol_results.append(True)
                sols.append(sol_candidate)
            is_singulars.append(False)
        return np.array(sol_results),np.array(is_singulars),np.array(sols)