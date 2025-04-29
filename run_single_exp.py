from utils.parser_utils import make_parser
import yaml

def run_single_exp(args):


    agent_name = args.agent_name


    with open(f'./agent_configs/{agent_name}.yaml') as f:
        agent_config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    env_name = args.env_name
    num_states = args.num_states
    num_agents = int(args.num_agents)

    num_features = args.num_features
    gamma = args.gamma


if __name__ == '__main__':


    args = make_parser()
