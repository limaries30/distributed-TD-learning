from single_exp import run_single_exp
from utils.parser_utils import make_parser
from multiprocessing import Pool
import numpy as np
import yaml
import copy


def run_single_exp_wrapper(args_config_save_dir):
    args,agent_config,save_dir = args_config_save_dir
    return run_single_exp(args,agent_config,save_dir)

def run_multiple_exp(args,agent_config,save_dir,num_processes=5):
    

    with Pool(processes=num_processes) as pool:
        arg_list = [(copy.deepcopy(args),agent_config,save_dir) for _ in range(num_processes)]
        results = pool.map(run_single_exp_wrapper,arg_list)
    
    error_collection = []
    for result in results:
        error_collection.append(result.logs["primal_error"])
    avg = np.mean(error_collection,axis=0)
    std = np.std(error_collection,axis=0)
    np.save(f'{save_dir}/primal_error_mean.npy',avg)
    np.save(f'{save_dir}/primal_error_std.npy',std)



        
if __name__ == '__main__':
    num_processes = 5
    args = make_parser()
    with open(f'./agents/configs/{args.agent_name}.yaml') as f:
        agent_config = yaml.load(f, Loader=yaml.FullLoader)
    run_multiple_exp(args,agent_config,num_processes)