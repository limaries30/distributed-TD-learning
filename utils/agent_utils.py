from agents.wang_elia import WangElia
from agents.gradient_tracking import GradientTracking
from agents.distributed_td import DistributedTD

def get_agent(agent_name:str,args,agent_config,graph):

    if agent_name=="WangElia":
        return WangElia(
                        num_states = args.num_states,
                        num_agents=args.num_agents,
                        num_features=args.num_features,
                        gamma = args.gamma,
                        agent_config = agent_config,
                        graph = graph
                        )
    if agent_name=="GradientTracking":
        return GradientTracking(
                                num_states = args.num_states,
                                num_agents=args.num_agents,
                                num_features=args.num_features,
                                gamma = args.gamma,
                                agent_config = agent_config,
                                graph = graph,
                                mixing_matrix_method=args.mixing_matrix_method
                               )       
    if agent_name=="DistributedTD":
        return DistributedTD(
                            num_states = args.num_states,
                            num_agents=args.num_agents,
                            num_features=args.num_features,
                            gamma = args.gamma,
                            agent_config = agent_config,
                            graph = graph,
                            mixing_matrix_method=args.mixing_matrix_method
                            )       