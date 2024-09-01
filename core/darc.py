import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.functional import relu, logsigmoid
from parameters import Parameters
from core import replay_memory
from core.mod_utils import is_lnorm_key
import numpy as np
from torch.distributions import Distribution, Normal

import numpy as np
import torch
from torch.nn import Module, Linear
from gym import spaces
import gym

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

class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std, device):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device),
                                      torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh

class Actor(nn.Module):
    
    def __init__(self, args, min_log_std=-20, max_log_std=2):
        super().__init__()

        self.args = args
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_action = args.max_action

        l1 = args.ls; l2 = args.ls; l3 = l2
        self.l1 = nn.Linear(self.state_dim, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l3, self.action_dim)	

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.args.device).unsqueeze(0)
        return self.forward(obs).detach().cpu().numpy()

    def evaluate(self, obs):
        obs = torch.FloatTensor(obs).to(self.args.device)
        return self.forward(obs), None, None, None, None
        
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



class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_action = args.max_action

        l1 = args.ls; l2 = args.ls; l3 = l2
        self.l1 = nn.Linear(self.state_dim + self.action_dim, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l3, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class DARC(object):
    def __init__(self, args):
        self.args = args
        self.max_action = 1.0
        self.device = args.device
        
        self.actor = Actor(args).to(self.device)
        self.actor_target = Actor(args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=6e-4)

        self.actor2 = Actor(args).to(self.device)
        self.actor2_target = Actor(args).to(self.device)
        self.actor2_optimizer = torch.optim.Adam(self.actor.parameters(), lr=6e-4)

        self.critic = Critic(args).to(self.device)
        self.critic_target = Critic(args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=6e-4)

        self.critic2 = Critic(args).to(self.device)
        self.critic2_target = Critic(args).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),lr=6e-4)

        self.log_alpha = torch.zeros((1,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # set target entropy to -|A|
        self.target_entropy = -args.action_dim
        
        self.discount = args.gamma
        self.tau = args.tau
        
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self,evo_times,all_fitness, all_gen , on_policy_states, on_policy_params, on_policy_discount_rewards,on_policy_actions,replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2, train_OFN_use_multi_actor= False, pop = None):
        actor_loss_list =[]
        critic_loss_list =[]
        pre_loss_list = []
        pv_loss_list = [0.0]
        keep_c_loss = [0.0]

        for it in range(iterations):
                
            x, y, u, r, d, _ ,_= replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            with torch.no_grad():
                next_action1 = self.actor_target(next_state)
                next_action2 = self.actor2_target(next_state)

                noise = torch.randn((action.shape[0], action.shape[1]), dtype=action.dtype, layout=action.layout, device=action.device) * noise_clip
                noise = noise.clamp(-noise_clip, noise_clip)

                next_action1 = (next_action1 + noise).clamp(-self.max_action, self.max_action)
                next_action2 = (next_action2 + noise).clamp(-self.max_action, self.max_action)

                next_Q1_a1 = self.critic_target(next_state, next_action1)
                next_Q2_a1 = self.critic2_target(next_state, next_action1)

                next_Q1_a2 = self.critic_target(next_state, next_action2)
                next_Q2_a2 = self.critic2_target(next_state, next_action2)

                next_Q1 = torch.min(next_Q1_a1, next_Q2_a1)
                next_Q2 = torch.min(next_Q1_a2, next_Q2_a2)

                next_Q = 0.1 * next_Q1 + 0.9 * next_Q2

                target_Q = reward + (done * discount * next_Q)

            epsilon = np.random.rand(0, 1)
            if epsilon < 0.5:
                current_Q1 = self.critic(state, action)
                current_Q2 = self.critic2(state, action)

                critic_loss1 = F.mse_loss(current_Q1, target_Q) + 0.1 * F.mse_loss(current_Q2, target_Q)

                self.critic_optimizer.zero_grad()
                critic_loss1.backward()
                self.critic_optimizer.step()

                actor_loss = -self.critic(state, self.actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_list.append(actor_loss.item())
                
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            else:
                current_Q1 = self.critic(state, action)
                current_Q2 = self.critic2(state, action)

                critic_loss2 = F.mse_loss(current_Q2, target_Q) + 0.1 * F.mse_loss(current_Q2, target_Q)

                self.critic2_optimizer.zero_grad()
                critic_loss2.backward()
                self.critic2_optimizer.step()

                actor2_loss = -self.critic2(state, self.actor2(state)).mean()
                
                self.actor2_optimizer.zero_grad()
                actor2_loss.backward()
                self.actor2_optimizer.step()
                actor_loss_list.append(actor2_loss.item())
                
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return np.mean(actor_loss_list) , np.mean(critic_loss_list), np.mean(pre_loss_list),np.mean(pv_loss_list), np.mean(keep_c_loss)



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
