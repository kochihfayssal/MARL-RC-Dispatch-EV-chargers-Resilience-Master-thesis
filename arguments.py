import argparse
import torch
from distutils.util import strtobool
import os 
import random 


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    parser.add_argument('--alg', type=str, default='mappo', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.90, help='the discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='the optimizer')
    parser.add_argument('--model_dir', type=str, default='D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/model',
                         help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='D:/_TUBERLIN/COURSES/__Master_Thesis/Code/Algorithms/SMAC/results',
                         help='the result directory of the policy')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--evaluate_cycle', type=int, default=10000, help='how often to evaluate the model')
    parser.add_argument('--limit_timesteps', type=int, default=24, help='maximal length of the episode')
    parser.add_argument('--n_steps', type=int, default=3000000, help='total time steps')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--gae_lambda", type=float, default=0.97, help="the lambda for the general advantage estimation")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="coefficient of the entropy")
    parser.add_argument("--max_grad_norm", type=float, default=10, help="the maximum norm for the gradient clipping")
    parser.add_argument("--max_kl", type=float, default=0.01, help="KL divergence constraint")
    parser.add_argument("--damping", type=float, default=0.1, help="Damping factor for conjugate gradient")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update rate for the target network")
    parser.add_argument("--ls_step", type=float, default=10, help="number of line search")
    parser.add_argument("--accept_ratio", type=float, default=0.5, help="accept ratio of loss improve")


    args = parser.parse_args()
    return args

def get_mixer_args(args):
    args.use_gpu = torch.cuda.is_available()
    args.rnn_hidden_dim = 64
    args.save_cycle = 9000
    args.n_episodes = 32
    args.ppo_n_epochs = 15
    return args