from run_single_exp import run_single_exp
from utils.parser_utils import make_parser
from multiprocessing import Pool
import numpy as np
if __name__ == '__main__':
    num_processes = 5
    args = make_parser()
    pool=Pool(processes=num_processes)
    results = pool.map(run_single_exp,[args for i in range(num_processes)])
    
    avg = []
    for result in results:
        avg.append(result.logs["total_error"])
    avg = np.mean(avg,axis=0)
    np.save(f'{args.save_dir}/{args.exp_id}/{args.num_agents}/total_error.npy',avg)
        