from agents.wang_elia import WangElia



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