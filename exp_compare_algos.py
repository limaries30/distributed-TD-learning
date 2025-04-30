from mutiple_exp import run_multiple_exp
from utils.parser_utils import make_parser
from multiprocessing import Pool
import numpy as np

import yaml


def run_algos():

    lrs = [1/(2**i) for i in range(3,7)]
    num_agents_list = [8,32]

    for lr in lrs:
        for num_agents in num_agents_list:
            args = make_parser()
            
            with open(f'./agents/configs/{args.agent_name}.yaml') as f:
                agent_config = yaml.load(f, Loader=yaml.FullLoader)
            args.num_agents = num_agents
            agent_config["alpha"]=lr
            save_dir = f'{args.log_root_dir}/{args.exp_id}/{args.num_agents}/{args.graph_type}/{agent_config["alpha"]}'
    
            run_multiple_exp(args,agent_config,save_dir=save_dir,num_processes=10)
    
        
if __name__ == '__main__':

    run_algos()