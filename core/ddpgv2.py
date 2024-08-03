import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from parameters import Parameters
from core import replay_memory
from core.mod_utils import is_lnorm_key
import numpy as np

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class GeneticAgent:
    def __init__(self, args: Parameters):

        self.args = args

        self.actor = Actor(args)
        self.old_actor = Actor(args)
        self.temp_actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.buffer = replay_memory.ReplayMemory(self.args.individual_bs, args.device)
        self.loss = nn.MSELoss()

    def keep_consistency(self, z_old, z_new):
        target_action = self.old_actor.select_action_from_z(z_old).detach()
        current_action = self.actor.select_action_from_z(z_new)
        delta = (current_action - target_action).abs()
        dt = torch.mean(delta ** 2)
        self.actor_optim.zero_grad()
        dt.backward()
        self.actor_optim.step()
        return dt.data.cpu().numpy()

    def keep_consistency_with_other_agent(self, z_old, z_new, other_actor):
        target_action = other_actor.select_action_from_z(z_old).detach()
        current_action = self.actor.select_action_from_z(z_new)
        delta = (current_action - target_action).abs()
        dt = torch.mean(delta ** 2)
        self.actor_optim.zero_grad()
        dt.backward()
        self.actor_optim.step()
        return dt.data.cpu().numpy()

    def update_parameters(self, batch, p1, p2, critic):
        state_batch, _, _, _, _ = batch

        p1_action = p1(state_batch)
        p2_action = p2(state_batch)
        p1_q = critic.Q1(state_batch, p1_action).flatten()
        p2_q = critic.Q1(state_batch, p2_action).flatten()

        eps = 0.0
        action_batch = torch.cat((p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        state_batch = torch.cat((state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps]))
        actor_action = self.actor(state_batch)

        # Actor Update
        self.actor_optim.zero_grad()
        sq = (actor_action - action_batch)**2
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()

class shared_state_embedding(nn.Module):
    def __init__(self, args):
        super(shared_state_embedding, self).__init__()
        self.args = args
        l1 = 400
        l2 = args.ls
        l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)
        # Init
        self.to(self.args.device)

    def forward(self, state):
        # Hidden Layer 1
        out = self.w_l1(state)
        if self.args.use_ln: out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = out.tanh()

        return out


class Actor(nn.Module):
    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        l1 = args.ls; l2 = args.ls; l3 = l2
        # Out
        self.w_out = nn.Linear(l3, args.action_dim)
        # Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, state_embedding):
        s_z = state_embedding.forward(input)
        action = self.w_out(s_z).tanh()
        return action

    def select_action_from_z(self,s_z):

        action = self.w_out(s_z).tanh()
        return action

    def select_action(self, state, state_embedding):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state, state_embedding).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(torch.sum((action_batch - self.forward(state_batch))**2, dim=-1))
        return novelty.item()

    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        self.norms = []
        self.hidden_sizes = hidden_sizes
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            norm = LayerNorm(next_size)
            self.add_module(f'fc{i}', fc)
            self.add_module(f'norm{i}', norm)
            self.fcs.append(fc)
            self.norms.append(norm)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.mul_(0.1)
        self.last_fc.bias.data.mul_(0.1)

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = F.leaky_relu(self.norms[i](fc(h)))
        output = self.last_fc(h)
        return output

class Critic(nn.Module):
    def __init__(self, args, n_nets=4):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(args.state_dim + args.action_dim, [512, 512, 512], 1)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action, different=False):
        sa = torch.cat((state, action), dim=-1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles




class Policy_Value_Network(nn.Module):
    def __init__(self, args, n_nets=4):
        super().__init__()
        self.st_nets = []
        self.nets = []
        self.args = args
        self.n_nets = n_nets
        if self.args.OFF_TYPE == 1 :
            input_dim = self.args.state_dim + self.args.action_dim
        else:
            input_dim = self.args.ls
        for i in range(n_nets):
            st_net = Mlp(self.args.ls + 1, [self.args.pr, self.args.pr], self.args.pr)
            self.add_module(f'st_qf{i}', st_net)
            self.st_nets.append(st_net)
            net = Mlp(input_dim + self.args.pr, [512, 512], 1)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, input, param):
        reshape_param = param.reshape([-1, self.args.ls + 1])
        out = []
        for i in range(self.n_nets):
            out_p = self.st_nets[i](reshape_param)
            out_p = out_p.reshape([-1, self.args.action_dim, self.args.pr])
            out_p = torch.mean(out_p, dim=1)
            concat_input = torch.cat((input, out_p), 1)
            out.append(self.nets[i](concat_input))
        return torch.stack(out, dim=1)

import random

def caculate_prob(score):

    X = (score - np.min(score))/(np.max(score)-np.min(score) + 1e-8)
    max_X = np.max(X)

    exp_x = np.exp(X-max_X)
    sum_exp_x = np.sum(exp_x)
    prob = exp_x/sum_exp_x
    return prob

class DDPG(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def td_error(self, state, action, next_state, reward, done):
        next_action = self.actor_target.forward(next_state)
        next_q = self.critic_target(next_state, next_action)

        done = 1 if done else 0
        if self.args.use_done_mask: next_q = next_q * (1 - done)  # Done mask
        target_q = reward + (self.gamma * next_q)

        current_q = self.critic(state, action)
        dt = (current_q - target_q).abs()
        return dt.item()

    def update_parameters(self, batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        # Load everything to GPU if not already
        self.actor_target.to(self.args.device)
        self.critic_target.to(self.args.device)
        self.critic.to(self.args.device)
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)

        # Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(next_state_batch, next_action_batch)
        if self.args.use_done_mask: next_q = next_q * (1 - done_batch) #Done mask
        target_q = reward_batch + (self.gamma * next_q).detach()

        self.critic_optim.zero_grad()
        current_q = self.critic.forward(state_batch, action_batch)
        delta = (current_q - target_q).abs()
        dt = torch.mean(delta**2)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()

        policy_grad_loss = -(self.critic.forward(state_batch, self.actor.forward(state_batch))).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()

def fanin_init(size, fanin=None):
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def actfn_none(inp): return inp

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class OUNoise:

    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
