from abc import ABC,abstractmethod

class AbstractAgent(ABC):

    def __init__(self,num_states:int,num_agents:int):
        self.num_states = num_states
        self.num_agents = num_agents


    @abstractmethod
    def update(self,current_state,next_state,reward,info):
        '''
        update learning parameters
        '''
        
        pass
