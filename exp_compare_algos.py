from mutiple_exp import run_multiple_exp
from utils.parser_utils import make_parser
from multiprocessing import Pool
import numpy as np

import yaml


def run_algos():

    num_agents_list = [8,32]
    agent_names = ["GradientTracking","WangElia","DistributedTD"]

    for agent_name in agent_names:
        for num_agents in num_agents_list:
            args = make_parser()
            args.agent_name = agent_name
            with open(f'./agents/configs/{args.agent_name}.yaml') as f:
                agent_config = yaml.load(f, Loader=yaml.FullLoader)
            args.num_agents = num_agents
            save_dir = f'{args.log_root_dir}/{args.exp_id}/{args.num_agents}/{args.graph_type}/{agent_name}/{agent_config["alpha"]}'
            if agent_name in ["GradientTracking","DistributedTD"]:
                save_dir = f'{save_dir}/{args.mixing_matrix_method}'
                
    
            run_multiple_exp(args,agent_config,save_dir=save_dir,num_processes=10)
    
        
if __name__ == '__main__':

    run_algos()