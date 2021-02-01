import math
import numpy as np
import argparse
import random

import torch as t
import torch.nn as nn
from torch.distributions import Normal, Laplace

import gym
import bayesfunc as bf

from buffer import BufferDiscrete
from moso import MO, SO, OneHotConcatSONet

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
parser.add_argument('--hidden_units',               type=int,   nargs='?', default=50)
parser.add_argument('--hidden_layers',              type=int,   nargs='?', default=0)
parser.add_argument('--train_batch',                type=int,   nargs='?', default=1024)
parser.add_argument('--gi_inducing_batch',          type=int,   nargs='?', default=200)
parser.add_argument('--gi_prec',                    type=str,   nargs='?', default='full',       choices=['full', 'diag'])
parser.add_argument('--gi_inducing_scale',          type=float, nargs='?', default=2.)
parser.add_argument('--S_eval',                     type=int,   nargs='?', default=100)
parser.add_argument('--S_train',                    type=int,   nargs='?', default=3)
parser.add_argument('--S_explore',                  type=int,   nargs='?', default=1)
parser.add_argument('--gamma',                      type=float, nargs='?', default=0.99)
parser.add_argument('--seed',                       type=int,   nargs='?', default=0)
parser.add_argument('--epsilon',                    type=float, nargs='?', default=1)
parser.add_argument('--train_steps_per_transition', type=int,   nargs='?', default=100)
parser.add_argument('--output_scale',               type=float, nargs='?', default=200.)
parser.add_argument('--action_weight',              type=float, nargs='?', default=0.1)
parser.add_argument('--state_weight',               type=float, nargs='?', default=1.)
args = parser.parse_args()


t.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)



def obs_shape(env):
    return env.observation_space.shape
def actions(env):
    return env.action_space.n

def mo_linear(env, args):
    shape = obs_shape(env)
    assert 1 == len(shape)
    in_features = shape[0]
    out_features = actions(env)
    nn = net(in_features, out_features+1, args)
    return MO(nn, actions(env), args)

def so_linear(env, args):
    shape = obs_shape(env)
    assert 1 == len(shape)
    in_features = shape[0] + actions(env)
    out_features = 1
    _net = net(in_features, out_features, args)
    return SO(OneHotConcatSONet(_net, actions(env)), actions(env), args)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = t.tensor(scale)
    def forward(self, x):
        return self.scale*x

def net(in_features, out_features, args):
    lin = {
        'det' : bf.DetLinear,
        'fac' : bf.FactorisedLinear,
        'gi'  : bf.GILinear
    }[args.vi_family]

    kwargs = {
        'det' : {},
        'fac' : {},
        'gi' : {'inducing_batch' : args.gi_inducing_batch, 'full_prec': args.gi_prec=='full'}
    }[args.vi_family]

    _net = nn.Sequential(
        lin(in_features, args.hidden_units, **kwargs),
        nn.ReLU(),
        *[nn.Sequential(lin(args.hidden_units, args.hidden_units, **kwargs), nn.ReLU()) for _ in range(args.hidden_layers)],
        lin(args.hidden_units, out_features, **kwargs),
        Scale(args.output_scale)
    )

    if args.vi_family == 'gi':
        ib = args.gi_inducing_batch
        _net = bf.InducingWrapper(_net, inducing_batch=ib, inducing_data=args.gi_inducing_scale*t.randn(ib, in_features))
    return _net

class AbstractPr(nn.Module):
    def forward(self, x):
        return Normal(x, self.sigma)
class FixedSigmaPr(AbstractPr):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
class LearnedSigmaPr(AbstractPr):
    def __init__(self, sigma):
        super().__init__()
        self.log_sigma = nn.Parameter(t.tensor(math.log(sigma)))
    @property
    def sigma(self):
        return self.log_sigma.exp()

class QLearning(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.env = gym.make(args.env)
        self.env.action_space.seed(args.seed)
        self.env.seed(args.seed)

        moso = {
            'mo' : mo_linear,
            'so' : so_linear,
        }[args.moso]

        Pr = {
            'fixed' : FixedSigmaPr,
            'learned' : LearnedSigmaPr,
        }[args.fixed_learned_sigma]

        self.Pr = Pr(args.sigma)

        self.moso = moso(self.env, args)
        dtype = getattr(t, args.dtype)
        self.buff = BufferDiscrete(obs_shape(self.env), args.storage_device, args.device, dtype)

        self.to(device=args.device, dtype=dtype)
        self.opt  = getattr(t.optim, args.opt)(self.parameters(), lr=args.lr)


    def eval_rollout(self, render=False):
        s = self.env.reset()
        d = False
        total_reward = 0.
        t = 0
        while not d:
            t += 1
            if render:
                self.env.render()
            Qs, _ = self.moso.Qs(s, S=args.S_eval)
            a = Qs.mean(0).argmax(-1).item() 
            s, r, d, info = self.env.step(a)
            total_reward += r
        return total_reward

    def train_step(self):
        batch_fun = {
            'random' : self.buff.random_batch,
            'contig' : self.buff.contiguous_batch
        }[self.args.random_contig]
        s, a, sp, r, d = batch_fun(self.args.train_batch)
        batch = s.shape[0]

        Qsa, Qsp, logpq = self.moso.Qsasp(s, a, sp, S=args.S_train)
        Qsp = Qsp.max(-1, keepdim=True)[0]
        if self.args.direct_residual == "direct":
            Qsp = Qsp.detach()
        bellman_error = Qsa - self.args.gamma*Qsp*(1-d)
        Pr = self.Pr(bellman_error)
        lpr = Pr.log_prob(r)
        assert lpr.shape == (args.S_train, batch, 1)
        lpr = lpr.mean((-1, -2))
        obj = lpr + logpq / self.buff.filled_buffer
        assert obj.shape == (args.S_train,)
        
        self.opt.zero_grad()
        (-obj.mean()).backward()
        self.opt.step()
        

    def train_rollout(self, epsilon=None):
        if epsilon is None:
            epsilon = self.args.epsilon

        s = self.env.reset()
        if   self.args.thompson_epi_trans == "episode":
            _, sample_dict = self.moso.Qs(s, S=self.args.S_explore)
        elif self.args.thompson_epi_trans == "transition":
            sample_dict = None

        d = False
        t = 0
        total_reward = 0.
        while (not d) and (t < self.env._max_episode_steps):
            if np.random.rand() < epsilon:
                a = self.env.action_space.sample()
            else:
                Qs, _ = self.moso.Qs(s, S=self.args.S_explore, sample_dict=sample_dict)
                a = Qs.mean(0).argmax(-1).item() 
            sp, r, d, info = self.env.step(a)

            if (not self.args.zero_final_q) and (t+1 == self.env._max_episode_steps):
                assert d
                d = False

            total_reward += r
            self.buff.add_state(s, a, sp, r, d)

            for _ in range(self.args.train_steps_per_transition):
                self.train_step()

            t+=1
            s = sp
        return total_reward
        

if __name__ == "__main__":
    q = QLearning(args)
    for i in range(100):
        q.train_rollout()
        print(f"episode: {i}; eval:  {q.eval_rollout()}; sigma: {q.Pr.sigma}; weight: {q.moso.action_weight}")
