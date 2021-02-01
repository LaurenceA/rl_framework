import argparse

import random
import numpy
import torch as t

from agent import Agent

#--vi_family det
#--vi_family gi --sigma 1. --epsilon 1. --output_scale 50
#--vi_family gi --S_eval 300 --epsilon 0.1 --sigma 1. --fixed_learned_sigma learned --output_scale 50.

parser = argparse.ArgumentParser()
parser.add_argument('output_filename',              type=str,   nargs='?', default='test')
parser.add_argument('--env',                        type=str,   nargs='?', default='CartPole-v0')
parser.add_argument('--moso',                       type=str,   nargs='?', default='mo',         choices=['mo', 'so'])
parser.add_argument('--vi_family',                  type=str,   nargs='?', default='fac',        choices=['det', 'fac', 'gi'])
parser.add_argument('--direct_residual',            type=str,   nargs='?', default='residual',   choices=['direct', 'residual'])
parser.add_argument('--random_contig',              type=str,   nargs='?', default='contig',     choices=['random', 'contig'])
parser.add_argument('--fixed_learned_sigma',        type=str,   nargs='?', default='fixed',      choices=['fixed', 'learned'])
parser.add_argument('--thompson_epi_trans',         type=str,   nargs='?', default='episode',    choices=['episode', 'transition'])
parser.add_argument('--zero_final_q',               action='store_true')
parser.add_argument('--sigma',                      type=float, nargs='?', default=0.1)
parser.add_argument('--device',                     type=str,   nargs='?', default='cpu',        choices=['cpu', 'cuda'])
parser.add_argument('--storage_device',             type=str,   nargs='?', default='cpu',        choices=['cpu', 'cuda'])
parser.add_argument('--dtype',                      type=str,   nargs='?', default='float64',    choices=['float32', 'float64'])
parser.add_argument('--opt',                        type=str,   nargs='?', default='Adam',)
parser.add_argument('--lr',                         type=float, nargs='?', default=0.01)
parser.add_argument('--beta',                       type=float, nargs='?', default=1.)
parser.add_argument('--hidden_units',               type=int,   nargs='?', default=50)
parser.add_argument('--hidden_layers',              type=int,   nargs='?', default=1)
parser.add_argument('--train_batch',                type=int,   nargs='?', default=1024)
parser.add_argument('--gi_inducing_batch',          type=int,   nargs='?', default=200)
parser.add_argument('--gi_prec',                    type=str,   nargs='?', default='full',       choices=['full', 'diag'])
parser.add_argument('--gi_inducing_scale',          type=float, nargs='?', default=2.)
parser.add_argument('--S_eval',                     type=int,   nargs='?', default=100)
parser.add_argument('--S_train',                    type=int,   nargs='?', default=3)
parser.add_argument('--S_explore',                  type=int,   nargs='?', default=1)
parser.add_argument('--gamma',                      type=float, nargs='?', default=0.99)
parser.add_argument('--seed',                       type=int,   nargs='?', default=0)
parser.add_argument('--epsilon',                    type=float, nargs='?', default=0.1)
parser.add_argument('--random_episodes',            type=int,   nargs='?', default=2)
parser.add_argument('--episodes',                   type=int,   nargs='?', default=100)
parser.add_argument('--train_steps_per_transition', type=int,   nargs='?', default=100)
parser.add_argument('--output_scale',               type=float, nargs='?', default=50.)
parser.add_argument('--action_weight',              type=float, nargs='?', default=0.1)
parser.add_argument('--state_weight',               type=float, nargs='?', default=1.)
args = parser.parse_args()


t.manual_seed(args.seed)
numpy.random.seed(args.seed)
random.seed(args.seed)


if __name__ == "__main__":
    agent = Agent(args)
    for i in range(args.random_episodes):
        agent.train_rollout(epsilon=1.)
        print(f"episode: {i}; eval:  {agent.eval_rollout()}; sigma: {agent.Pr.sigma}")
    for i in range(args.episodes):
        agent.train_rollout()
        print(f"episode: {i}; eval:  {agent.eval_rollout()}; sigma: {agent.Pr.sigma}")
