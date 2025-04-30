from mutiple_exp import run_mutiple_exp
from utils.parser_utils import make_parser
from multiprocessing import Pool
import numpy as np

import yaml


def run_lr_num_agents():

    lrs = [0.1,0.01]
    num_agents_list = [10,100]

    for lr in lrs:
        for num_agents in num_agents_list:
            args = make_parser()
            with open(f'./agents/configs/{args.agent_name}.yaml') as f:
                agent_config = yaml.load(f, Loader=yaml.FullLoader)
            run_mutiple_exp(args,agent_config)
    
        
if __name__ == '__main__':

    run_lr_num_agents()