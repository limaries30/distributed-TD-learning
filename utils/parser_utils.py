import argparse


def make_parser():




    # 입력받을 인자값 등록
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--prefix',type=str, required=False )
    parser.add_argument('--second_prefix', type=str,required=False)
    parser.add_argument('--prefix_value',type=float ,required=False)
    parser.add_argument('--second_prefix_value',type=float, required=False)
    parser.add_argument('--save_dir_suffix',type=str, required=False)
    


    
    parser.add_argument('--exp_id',default="7cc8c", type=str)
    parser.add_argument('--agent_name',type=str,default="WangElia")
    parser.add_argument('--env_name',type=str,default="DistributedMDP_1")
    parser.add_argument('--num_states',type=int,default=3)

    parser.add_argument('--num_agents',type=int,default=8)
    parser.add_argument('--total_steps',type=int,default=int(2*1e5))

    parser.add_argument('--num_features',type=int,default=2)
    parser.add_argument('--feature_name',type=str,default="fourier_feature")

    parser.add_argument('--gamma',type=float,default=0.8)
    parser.add_argument('--graph_type',type=str,default="ring")

    parser.add_argument('--print_freq',type=int,default=100)
    parser.add_argument('--save_dir',type=str,default="./results")



    # 입력받은 인자값을 args에 저장 (type: namespace)
    args = parser.parse_args()

    return args
