import torch as t
import torch.nn as nn
import bayesfunc as bf
from buffer import BufferDiscrete

def one_hot(a, actions):
    """
    Utility function that converts the integer tensor, a, of shape [T, 1]
    to a one-hot representation of shape [T, actions]
    """
    assert 1 == a.shape[-1]
    return (a == t.arange(actions, device=a.device)).to(dtype=t.float)

def expand_unsqueeze_0(tensor, S):
    """
    Introduces a new zeroth dimension of size S
    """
    return tensor.expand(S, *(len(tensor.shape)*[-1]))

class MOSO(nn.Module):
    def __init__(self, net, actions, args):#, actions, device='cpu', dtype=t.float32):
        super().__init__() 
        self.net = net
        self.actions = actions
        self.device = args.device
        self.dtype = getattr(t, args.dtype)
        self.action_weight = args.action_weight
        self.state_weight = args.state_weight

class MO(MOSO):
    #Input states as [T, F]
    #Output must be [S, T, A]
    def Qs(self, s, S=1, sample_dict=None):
        """
        Exclusively for computing q(s, a) for all actions for one state for rollouts
        """
        assert 1 == len(s.shape)
        s = t.tensor(s, device=self.device, dtype=self.dtype).unsqueeze(0)
        output, _, output_sample_dict = self._Qs(s, S, sample_dict)
        return output, output_sample_dict
        

    def _Qs(self, s, S=1, sample_dict=None):
        """
        Computes q(s, a) for the input states, for all actions.
        output:
        output
        logpq (0. for a non-Bayesian net).
        sample_dict (for fixing the weights).
        """
        s = expand_unsqueeze_0(s, S)
        output, logpq, sample_dict = bf.propagate(self.net, s, sample_dict=sample_dict)
        output = self.state_weight * output[..., 0:1] + self.action_weight * output[..., 1:]
        return output, logpq, sample_dict

    def Qsasp(self, s, a, sp, S=1):
        """
        Returns q(s_t, a_t), q(s_t', a) for all a, and the regulariser logpq, for training
        """
        s = s.to(device=self.device, dtype=self.dtype)
        a = a.to(device=self.device, dtype=self.dtype)

        T = s.shape[0]
        assert s.shape[0] == sp.shape[0]
        ssp = t.cat([s, sp], 0)
        Qssp, logpq, _ = self._Qs(ssp, S=S)
        if a.device == t.device('cpu'):
            # gather on cpu wants 64 bit integers
            a = a.to(dtype=t.int64)
        Qsa = Qssp.gather(-1, a.unsqueeze(0))
        Qsp = Qssp[:, T:, :]
        return Qsa, Qsp, logpq
        
        
class SO(MOSO):
    #Input states and actions as 
    #Output must be [S, T, 1]
    def Qsa(self, s, a, S=1, sample_dict=None):
        s = expand_unsqueeze_0(s, S)
        a = expand_unsqueeze_0(a, S)
        return bf.propagate(self.net, s, a, sample_dict=sample_dict)

    def combine_all_actions(self, s):
        T = s.shape[0]
        feature_shape = s.shape[1:]
        #[1, A]
        a = t.arange(self.actions, device=s.device)[None, :]
        #[T, A]
        a = a.expand(T, self.actions)
        #[TA, 1]
        a = a.reshape(T*self.actions, 1)
        #[T, A, ...]
        s = s.unsqueeze(1).expand(T, self.actions, *feature_shape)
        #[TA, ...]
        s = s.reshape(T*self.actions, *feature_shape)
        return s, a

    def Qs(self, s, S=1, sample_dict=None):
        """
        Exclusively for computing q(s, a) for all actions for one state for rollouts
        """
        assert 1 == len(s.shape)
        s = t.tensor(s, device=self.device, dtype=self.dtype).unsqueeze(0)
        T = s.shape[0]

        s, a = self.combine_all_actions(s)
        #Qs: [S, TA, 1]
        Qs, _, output_sample_dict = self.Qsa(s, a, S=S, sample_dict=sample_dict)
        #Qs: [S, T, A]
        return Qs.view(S, T, self.actions), output_sample_dict
        

    def Qsasp(self, s, a, sp, S=1):
        s = t.tensor(s, device=self.device, dtype=self.dtype)
        a = t.tensor(a, device=self.device, dtype=self.dtype)

        T = s.shape[0]
        assert T ==  a.shape[0] 
        assert T == sp.shape[0] 

        _s, _a = self.combine_all_actions(sp)
        _s = t.cat([_s, s], 0)
        _a = t.cat([_a, a], 0)
 
        Qs, logpq, output_sample_dict = self.Qsa(_s, _a, S=S)
        Qsa = Qs[:, -T:]
        Qsp = Qs[:, :-T].view(S, T, self.actions)
        
        return Qsa, Qsp, logpq


class OneHotConcatSONet(nn.Module):
    """
    takes a neural network with a single input tensor of shape [T, S+A], where
    S is the dimension of the state space and A is the number of discrete actions,
    and converts it to a network that takes two inputs:
    states as a floating-point matrix
    s: [T, S]
    and actions as an integer vector,
    a: [T, 1]
    """
    def __init__(self, net, actions):
        super().__init__()
        self.net = net
        self.actions = actions

    def forward(self, s, a):
        a = one_hot(a, self.actions)
        sa = t.cat([s, a], -1)
        return self.net(sa)

if __name__ == "__main__":
    state_features = 2
    hiddens = 50
    actions = 3
    T = 100
    mo_net = nn.Sequential(
       nn.Linear(2, 50),
       nn.ReLU(),
       nn.Linear(50, 3)
    )
    so_net = nn.Sequential(
       nn.Linear(5, 50),
       nn.ReLU(),
       nn.Linear(50, 1)
    )
    so_net = OneHotConcatSONet(so_net, actions)

    _s = t.randn(T, state_features)
    a = t.randint(actions, (T, 1))

    mo = MO(mo_net)
    so = SO(so_net, actions) 
