from utils.parser_utils import make_parser
from utils.graph_utils import get_graph
from utils.mdp_utils import get_mdp
from utils.agent_utils import get_agent
from logging_module import Logger
import yaml


def run_single_exp(args,agent_config,save=True)->Logger:

    '''
        save : save the loggre file
    '''

    logger = Logger(save_dir=f'{args.save_dir}/{args.exp_id}/{args.num_agents}')


    env_name = args.env_name
    num_agents = int(args.num_agents)
    
    graph_type = args.graph_type
    graph = get_graph(graph_type,num_agents)  # Characterizes the connection of the agents

    agent_name = args.agent_name
    agent = get_agent(args.agent_name,args,agent_config,graph)
    
    env = get_mdp(env_name,args,graph)
    total_steps = args.total_steps

    current_state = env.reset()
    for steps in range(total_steps):

        primal_error = env.calc_primal_error(agent.bar_theta)
        dual_error =   env.calc_dual_error(agent.bar_w)
        logger.add("total_error",primal_error+dual_error)


        next_state,rewards,info = env.step()

        agent.update(current_state,next_state,rewards,info)

        current_state = next_state


        
        if steps%args.print_freq==0:
            print(f'steps: {steps}')
            print('primal_error',primal_error)
            print('dual_error',dual_error)
    
    if save:
        logger.save_logs()
    return logger
        

if __name__ == '__main__':


    args = make_parser()
    with open(f'./agents/configs/{args.agent_name}.yaml') as f:
        agent_config = yaml.load(f, Loader=yaml.FullLoader)
    run_single_exp(args,agent_config)
