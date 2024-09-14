import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.autograd import grad
from parameters import Parameters
from core import replay_memory
from core.mod_utils import is_lnorm_key
import numpy as np
from torch.distributions import Normal

torch.autograd.set_detect_anomaly(True)

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
        self.max_action = args.max_action
        l1 = args.ls; l2 = args.ls; l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, args.action_dim)

        # Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input):

        # Hidden Layer 1
        out = self.w_l1(input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = out.tanh()

        # Out
        out = (self.w_out(out)).tanh()
        return out

    def evaluate(self, state):
        action = self.forward(state)
        log_prob = Normal(0, 1).log_prob(action).sum(dim=-1)
        return action * self.max_action, log_prob

    def select_action_from_z(self,s_z):
        action = self.forward(s_z)
        return action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()

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

class Q_network(nn.Module):
    
    def __init__(self, args, l1=400, l2=300, l3=300):
        super(Q_network, self).__init__()
        self.args = args
        self.l1 = l1
        # Construct input interface (Hidden Layer 1)
        self.w_l1 = nn.Linear(args.state_dim+args.action_dim, l1)
        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln:
            self.lnorm1 = LayerNorm(l1)
            self.lnorm2 = LayerNorm(l2)
        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)
        self.to(self.args.device)

    def forward(self, input_):
        # Hidden Layer 1 (Input Interface)
        out = self.w_l1(input_)
        if self.args.use_ln:out = self.lnorm1(out)
        out = F.leaky_relu(out)
        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)
        # Output interface
        out = self.w_out(out)
        return out

class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 400;
        l2 = 300;
        l3 = l2

        # Construct input interface (Hidden Layer 1)
        self.w_l1 = nn.Linear(args.state_dim+args.action_dim, l1)
        # Hidden Layer 2

        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln:
            self.lnorm1 = LayerNorm(l1)
            self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.w_l3 = nn.Linear(args.state_dim+args.action_dim, l1)
        # Hidden Layer 2
        self.w_l4 = nn.Linear(l1, l2)
        if self.args.use_ln:
            self.lnorm3 = LayerNorm(l1)
            self.lnorm4 = LayerNorm(l2)

        # Out
        self.w_out_2 = nn.Linear(l3, 1)
        self.w_out_2.weight.data.mul_(0.1)
        self.w_out_2.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, action):

        # Hidden Layer 1 (Input Interface)
        concat_input = torch.cat([input,action],-1)

        out = self.w_l1(concat_input)
        if self.args.use_ln:out = self.lnorm1(out)

        out = F.leaky_relu(out)
        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)
        # Output interface
        out_1 = self.w_out(out)

        out_2 = self.w_l3(concat_input)
        if self.args.use_ln: out_2 = self.lnorm3(out_2)
        out_2 = F.leaky_relu(out_2)

        # Hidden Layer 2
        out_2 = self.w_l4(out_2)
        if self.args.use_ln: out_2 = self.lnorm4(out_2)
        out_2 = F.leaky_relu(out_2)

        # Output interface
        out_2 = self.w_out_2(out_2)

        return out_1, out_2

    def Q1(self, input, action):

        concat_input = torch.cat([input, action], -1)

        out = self.w_l1(concat_input)
        if self.args.use_ln:out = self.lnorm1(out)

        out = F.leaky_relu(out)
        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.leaky_relu(out)
        # Output interface
        out_1 = self.w_out(out)
        return out_1

class Ensemble_Critics(nn.Module):
    def __init__(self, args, n_nets = 4):
        super(Ensemble_Critics, self).__init__()
        self.args = args

        l1 = 400;
        l2 = 300;
        l3 = l2

        self.n_nets = n_nets
        self.critics = [
            Critic(args) for _ in range(n_nets)
        ]

    def forward(self, input, action):
        Q1_value = torch.tensor([]).to(self.args.device)
        Q2_value = torch.tensor([]).to(self.args.device)

        for critic in self.critics:
            current_Q1, current_Q2 = critic(input, action)
            Q1_value = torch.cat([Q1_value, current_Q1.unsqueeze(1)], 1).to(self.args.device)
            Q2_value = torch.cat([Q2_value, current_Q2.unsqueeze(1)], 1).to(self.args.device)

        return Q1_value, Q2_value

    def Q1(self, input, action):
        concat_input = torch.cat([input, action], -1)
        
        out_all = torch.tensor([]).to(self.args.device)
        for net in self.critics:
            out = net.w_l1(concat_input)
            if self.args.use_ln:out = net.lnorm1(out)

            out = F.leaky_relu(out)
            # Hidden Layer 2
            out = net.w_l2(out)
            if self.args.use_ln: out = net.lnorm2(out)
            out = F.leaky_relu(out)
            # Output interface
            out_1 = net.w_out(out)
            out_all = torch.cat([out_all, out_1.unsqueeze(1)], 1).to(self.args.device)
        return torch.max(out_all, 1)[0]


# class Critic(nn.Module):

#     def __init__(self, args, n_nets = 3):
#         super(Critic, self).__init__()
#         self.args = args

#         l1 = 400;
#         l2 = 300;
#         l3 = l2
        
#         self.Q1_nets = []
#         self.Q2_nets = []

#         for i in range(n_nets):
#             self.Q1_network = Q_network(args, l1, l2, l3)
#             self.Q2_network = Q_network(args, l1, l2, l3)

#             self.add_module('Q1_network_' + str(i), self.Q1_network)
#             self.add_module('Q2_network_' + str(i), self.Q2_network)
            
#             self.Q1_nets.append(self.Q1_network)
#             self.Q2_nets.append(self.Q2_network)

#         self.all_nets = self.Q1_nets + self.Q2_nets

#     def forward(self, state, action):
#         sa = torch.cat((state, action), dim=-1)
#         quantiles_Q1 = torch.stack(tuple(net(sa) for net in self.Q1_nets), dim=1)
#         quantiles_Q2 = torch.stack(tuple(net(sa) for net in self.Q2_nets), dim=1)
#         return quantiles_Q1, quantiles_Q2
        
        
#     def Q1(self, input, action):

#         out_all = torch.tensor([]).to(self.args.device)
#         for net in self.Q1_nets:
#             concat_input = torch.cat([input, action], -1)

#             out = net.w_l1(concat_input)
#             if self.args.use_ln:out = net.lnorm1(out)

#             out = F.leaky_relu(out)
#             # Hidden Layer 2
#             out = net.w_l2(out)
#             if self.args.use_ln: out = net.lnorm2(out)
#             out = F.leaky_relu(out)
#             # Output interface
#             out_1 = net.w_out(out)
#             out_all = torch.cat([out_all, out_1.unsqueeze(1)], 1).to(self.args.device)
        
#         return torch.max(out_all, 1)[0] 


def caculate_prob(score):

    X = (score - np.min(score))/(np.max(score)-np.min(score) + 1e-8)
    max_X = np.max(X)

    exp_x = np.exp(X-max_X)
    sum_exp_x = np.sum(exp_x)
    prob = exp_x/sum_exp_x
    return prob

class TD3(object):
    def __init__(self, args):
        self.args = args
        self.max_action = 1.0
        self.device = args.device
        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Ensemble_Critics(args, n_nets=3).to(self.device)
        self.critic_target = Ensemble_Critics(args, n_nets=3).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.critic_optimizer = [
            torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critic.critics
        ]

        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)


        # self.PVN = Policy_Value_Network(args).to(self.device)
        # self.PVN_Target = Policy_Value_Network(args).to(self.device)
        # self.PVN_Target.load_state_dict(self.PVN.state_dict())
        # self.PVN_optimizer = torch.optim.Adam([{'params': self.PVN.parameters()}],lr=1e-3)

        # self.state_embedding = shared_state_embedding(args)
        # self.state_embedding_target = shared_state_embedding(args)
        # self.state_embedding_target.load_state_dict(self.state_embedding.state_dict())
      
      
        # self.old_state_embedding = shared_state_embedding(args)
        # self.state_embedding_optimizer = torch.optim.Adam(self.state_embedding.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def approximate_underestimate(self, std_target_Q, replay_buffer):
        history_target_std = replay_buffer.update(self.args, std_target_Q)
        history_target_std = torch.FloatTensor(np.array(history_target_std)).unsqueeze(-1).to(self.device)
        history_target_std_std, _ = torch.std_mean(history_target_std[:, 1:], 1)
        under_target_std = torch.where(std_target_Q < 0.5, std_target_Q**1.1, std_target_Q)
        if self.args.bellman_mode == "TV":
            underestimate = torch.where((history_target_std_std < self.args.std_std_threshold), under_target_std, std_target_Q * 0)
        elif self.args.bellman_mode == "NV":
            underestimate = 0
        elif self.args.bellman_mode == "NT":
            underestimate = under_target_std
        
        return underestimate

    def train(self,evo_times,all_fitness, all_gen , on_policy_states, on_policy_params, on_policy_discount_rewards,on_policy_actions,replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2, train_OFN_use_multi_actor= False,all_actor = None):
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
                # Select action according to policy and add clipped noise
                noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, noise_clip)

                next_action = (self.actor_target.forward(next_state)+noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                quantiles_target_Q1, quantiles_target_Q2 = self.critic_target(next_state, next_action)
                # quantiles_target_Q = torch.min(quantiles_target_Q1, quantiles_target_Q2)

                std_target_Q1, mean_target_Q1 = torch.std_mean(quantiles_target_Q1, 1)
                std_target_Q2, mean_target_Q2 = torch.std_mean(quantiles_target_Q2, 1)

                underestimate = self.approximate_underestimate(torch.mean(std_target_Q1, std_target_Q2), replay_buffer)

                target_Q = reward + (done * discount * torch.min(mean_target_Q1, mean_target_Q2) + underestimate).detach()

            for critic, critic_target, critic_optimizer in zip(self.critic.critics, self.critic_target.critics, self.critic_optimizer):
                # Get current Q estimates
                current_Q1, current_Q2 = critic(state, action)
                # std_current_Q1, mean_current_Q1 = torch.std_mean(quantiles_current_Q1)
                # std_current_Q2, mean_current_Q2 = torch.std_mean(quantiles_current_Q2)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10)
                critic_optimizer.step()
                critic_loss_list.append(critic_loss.cpu().data.numpy().flatten())

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                s_z = state
                actor_loss = -1*torch.mean(critic.Q1(state, self.actor.select_action_from_z(s_z)))
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                self.actor_optimizer.step()

                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                actor_loss_list.append(actor_loss.cpu().data.numpy().flatten())
                pre_loss_list.append(0.0)
            # Get current Q estimates
            # quantiles_current_Q1, quantiles_current_Q2 = self.critic(state, action)
            
            # std_current_Q1, mean_current_Q1 = torch.std_mean(quantiles_current_Q1, 1)
            # std_current_Q2, mean_current_Q2 = torch.std_mean(quantiles_current_Q2, 1)

            # # Compute critic loss
            # critic_loss = F.mse_loss(mean_current_Q1, target_Q) + F.mse_loss(mean_current_Q2, target_Q)

            # # Optimize the critic
            # self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            # self.critic_optimizer.step()
            # critic_loss_list.append(critic_loss.cpu().data.numpy().flatten())
            
            # # Compute the target Q value
            # target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # target_Q = torch.min(target_Q1, target_Q2)
            # target_Q = reward + (done * discount * target_Q).detach()
            
            # # Get current Q estimates
            # current_Q1, current_Q2 = self.critic(state, action)

            # # Compute critic loss
            # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
 
            # # Optimize the critic
            # self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            # self.critic_optimizer.step()
            # critic_loss_list.append(critic_loss.cpu().data.numpy().flatten())

            # # Select action according to policy and add clipped noise
            # noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
            # noise = noise.clamp(-noise_clip, noise_clip)

            # next_action = (self.actor_target.forward(next_state)+noise).clamp(-self.max_action, self.max_action)

            # # Compute the target Q value
            # target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # std_target_Q1, mean_target_Q1 = torch.std_mean(target_Q1, 1)
            # std_target_Q2, mean_target_Q2 = torch.std_mean(target_Q2, 1)

            # underestimate_Q1 = self.approximate_underestimate(std_target_Q1, replay_buffer)
            # underestimate_Q2 = self.approximate_underestimate(std_target_Q2, replay_buffer)
            
            # std_target_Q, mean_target_Q = torch.std_mean(torch.stack([mean_target_Q1, mean_target_Q2], dim=1), 1)

            # # target_Q = torch.min(mean_target_Q1, mean_target_Q2)
            # target_Q = reward + (done * discount * torch.min(mean_target_Q1, mean_target_Q2)).detach()
            
            # # Get current Q estimates
            # current_Q1, current_Q2 = self.critic(state, action)

            # # _, mean_target_Q1 = torch.std_mean(current_Q1, 1)
            # # _, mean_target_Q2 = torch.std_mean(current_Q2, 1)

            # # Compute critic loss
            # critic_loss = F.mse_loss(torch.mean(current_Q1, dim=1), target_Q) + F.mse_loss(torch.mean(current_Q2, dim=1), target_Q)
            # # Optimize the critic
            # self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            # self.critic_optimizer.step()
            # critic_loss_list.append(critic_loss.cpu().data.numpy().flatten())

            # Delayed policy updates
            # if it % policy_freq == 0:

            #     # Compute actor loss
            #     s_z = state
            #     actor_loss = -1*torch.mean(self.critic.Q1(state, self.actor.select_action_from_z(s_z)))
            #     # Optimize the actor
            #     self.actor_optimizer.zero_grad()
            #     actor_loss.backward()
            #     nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            #     self.actor_optimizer.step()
                
            #     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            #     for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            #         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            #     actor_loss_list.append(actor_loss.cpu().data.numpy().flatten())
            #     pre_loss_list.append(0.0)

        return np.mean(actor_loss_list) , np.mean(critic_loss_list), np.mean(pre_loss_list),pv_loss_list, keep_c_loss



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