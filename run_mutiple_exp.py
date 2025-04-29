from run_single_exp import run_single_exp
from utils.parser_utils import make_parser


if __name__ == '__main__':
    num_runs = 5
    args = make_parser()
    logger = run_single_exp(args,save=False)
    for i in range(num_runs):
        