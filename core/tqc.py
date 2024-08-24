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
        self.action_dim = self.args.action_dim
        self.state_dim = self.args.state_dim
        self.max_action = args.max_action
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.net = Mlp(self.state_dim, [256, 256], 2 * self.action_dim)
        self.net.to(self.args.device)

    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std, self.args.device)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.args.device)[None, :]
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.forward(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(torch.tensor([0.] * self.args.action_dim), 1)    #actor dim=4

        z = noise.sample()
        action = torch.tanh(batch_mu + batch_sigma*z.to(self.args.device))
        log_prob = (dist.log_prob(batch_mu + batch_sigma * z.to(self.args.device)) -\
                   torch.log((1 - action.pow(2)) * self.max_action + 1e-7)).sum(-1).unsqueeze(dim=-1)
        return action * self.max_action, log_prob, z, batch_mu, batch_log_sigma

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
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output

class Critic(nn.Module):
    def __init__(self, args, n_nets=5, n_quantiles=25, top_quantiles_to_drop_per_net=2):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        self.n_quantiles = n_quantiles
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        for i in range(n_nets):
            net = Mlp(args.state_dim + args.action_dim, [512, 512, 512], 1)
            net.to(args.device)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action, different=False):
        sa = torch.cat((state, action), dim=-1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles

def caculate_prob(score):

    X = (score - np.min(score))/(np.max(score)-np.min(score) + 1e-8)
    max_X = np.max(X)

    exp_x = np.exp(X-max_X)
    sum_exp_x = np.sum(exp_x)
    prob = exp_x/sum_exp_x
    return prob

def quantile_huber_loss_f(quantiles, samples, device):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

class TQC(object):
    def __init__(self, args):
        self.args = args
        self.max_action = 1.0
        self.device = args.device
        self.actor = Actor(args).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=6e-4)

        self.critic = Critic(args, n_nets=args.n_nets).to(self.device)
        self.critic_target = Critic(args, n_nets=args.n_nets).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=6e-4)

        self.log_alpha = torch.zeros((1,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # set target entropy to -|A|
        self.target_entropy = -args.action_dim
        
        self.discount = args.gamma
        self.tau = args.tau

        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        self.top_quantiles_to_drop = self.critic.top_quantiles_to_drop_per_net * self.critic.n_nets
        
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

            # with torch.no_grad():
            sample_next_action, next_log_prob, z, batch_mu, batch_log_sigma = self.actor.evaluate(next_state)

            # compute and cut quantiles at the next action
            next_quantiles = self.critic_target(next_state, sample_next_action)
            sorted_next_quantiles, _ = torch.sort(next_quantiles.reshape(batch_size, -1))
            sorted_next_quantiles_part = sorted_next_quantiles[:, :self.quantiles_total - self.top_quantiles_to_drop]

            # compute target
            target_Q = reward + (done * discount * (sorted_next_quantiles_part - self.alpha * next_log_prob))

            cur_Q = self.critic(state, action)
            critic_loss = quantile_huber_loss_f(cur_Q, target_Q, self.device)
            
            # --- Update ---
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                 # --- Policy and alpha loss ---
                new_action, log_pi = self.actor(state)
                alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
                actor_loss = (self.args.alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # for param, target_param in zip(self.PVN.parameters(), self.PVN_Target.parameters()):
                #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                actor_loss_list.append(actor_loss.cpu().data.numpy().flatten())
                pre_loss_list.append(0.0)

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
