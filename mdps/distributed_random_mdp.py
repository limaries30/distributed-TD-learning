from mdps.abstract_distributed_mdp import AbstractDistributedMDP
import numpy as np
from utils.feature_utils import get_feature
from utils.mdp_utils import make_prob_matrix,make_random_behavior_policy_matrix
from typing import Tuple


class DistributedRandomMDP(AbstractDistributedMDP):

    def __init__( self, num_states:int, num_features:int,
                  gamma:float,num_agents:int,
                  feature_name:str,
                  graph:np.array
                ):


        # feature matrix size of (|S|,num_features)
        X = get_feature(feature_name)(num_states,num_features)
    
        # Transition matrix size of (|S|, |S|)
        P = make_prob_matrix(num_states,num_states)


        # Reward matrix size of ( N,|S|, |S|)
        barR = np.random.uniform(-1, 1, size=(num_agents, num_states , num_states)).round(2)


        
        super().__init__(num_states=num_states,
                         num_features =num_features, num_agents=num_agents, 
                         X=X,P= P, barR= barR, gamma=gamma)
        

        self.A = self.X.T@self.Dbeta@self.X-self.gamma * self.X.T@self.Dbeta@self.P@self.X
        self.barA = np.kron(np.eye(self.num_agents),self.A)


        avgR = np.average(self.barR,axis=0)
        expR = np.sum(np.multiply(self.P,avgR),axis=1)
        self.theta_sol = np.linalg.pinv(self.A)@self.X.T@self.Dbeta@expR
        self.bar_theta_sol = np.kron(np.ones(num_agents),self.theta_sol)[:,np.newaxis] # (num_agents*num_features,1)


        self.laplacian_matrix = graph
        self.bar_laplacian = np.kron(self.laplacian_matrix,np.eye(self.num_features)) #(num_agents*num_features,num_agents*num_features)
        self.bar_lap_pinv = np.linalg.pinv(self.bar_laplacian)
        
        barb = np.hstack([self.X.T@self.Dbeta@np.sum(np.multiply(self.P,self.barR[i,:,:]),axis=1)  for i  in range(self.num_agents)])[:,np.newaxis]
        self.Lw_sol =  -self.barA@self.bar_theta_sol+barb


    def calc_primal_error(self,theta):
        error = 1/self.num_agents * np.sum(np.square(self.bar_theta_sol-theta))
        return error

    def calc_dual_error(self,w):
  
        dual_error = 1/self.num_agents *np.sum(np.square(self.bar_laplacian@self.bar_lap_pinv@w-self.Lw_sol))
        return dual_error


    def reset(self):
        self.current_state_idx = np.random.randint(0,self.num_states) 
        return self.X[self.current_state_idx,:]
    
    def step(self)->Tuple[np.array,np.array,dict]:
        '''
        return [(d,),(N,),dict]
        '''

        next_state_idx = np.random.choice(self.num_states,1,p=self.P[self.current_state_idx,:])[0]
        next_state = self.X[next_state_idx,:]
        rewards = self.barR[:,self.current_state_idx,next_state_idx] #size of (N,)
        self.current_state_idx = next_state_idx
        info = {}
        
        return next_state,rewards,info