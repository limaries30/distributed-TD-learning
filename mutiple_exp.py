from single_exp import run_single_exp
from utils.parser_utils import make_parser
from multiprocessing import Pool
import numpy as np
import yaml
import copy


def run_single_exp_wrapper(args_and_config):
    args,agent_config = args_and_config
    return run_single_exp(args,agent_config)

def run_multiple_exp(args,agent_config,num_processes=5):


    with Pool(processes=num_processes) as pool:
        arg_list = [(copy.deepcopy(args),agent_config) for _ in range(num_processes)]
        results = pool.map(run_single_exp_wrapper,arg_list)
    
    avg = []
    for result in results:
        avg.append(result.logs["total_error"])
    avg = np.mean(avg,axis=0)
    np.save(f'{args.save_dir}/{args.exp_id}/{args.num_agents}/{agent_config["alpha"]}/total_error.npy',avg)



        
if __name__ == '__main__':
    num_processes = 5
    args = make_parser()
    with open(f'./agents/configs/{args.agent_name}.yaml') as f:
        agent_config = yaml.load(f, Loader=yaml.FullLoader)
    run_multiple_exp(args,agent_config,num_processes)