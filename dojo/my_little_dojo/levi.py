import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

learning_rate = 0.001
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 2
dim_act = 3


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(6, 64)  # opponent's action and my action, one-hot encoded
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32 + 1, dim_act)  # concat time
        self.fc_v = nn.Linear(32 + 1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x, t = x.view(-1, 7)[:, :-1], x.view(-1, 7)[:, -1]
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)  # (seq_len, batch_size, input_size)
        x, lstm_hidden = self.lstm(x, hidden)
        logits = self.fc_pi(torch.cat([x, t.view(-1, 1, 1)], dim=2))
        return logits, lstm_hidden

    def v(self, x, hidden):
        x, t = x.view(-1, 7)[:, :-1], x.view(-1, 7)[:, -1]
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(torch.cat([x, t.view(-1, 1, 1)], dim=2))
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, logits_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, logits, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            logits_lst.append(logits)
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, logits = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(logits_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, logits, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, pi_k_logits, (h_in, c_in), (h_out, c_out) = self.make_batch()
        first_hidden = (h_in.detach(), c_in.detach())  # each (1, 1, dim_h)
        second_hidden = (h_out.detach(), c_out.detach())  # each (1, 1, dim_h)

        pi_k = F.softmax(pi_k_logits, dim=1).view(-1, dim_act)
        pi_k_a = pi_k.gather(1, a)

        kl_div_history = []
        loss_pi_history = []
        loss_v_history = []
        for i in range(K_epoch):
            # update target with new V at each PPO update, to mitigate stale targets/
            v_prime = self.v(s_prime, second_hidden).view(-1, 1)  # (T, 1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).view(-1, 1)
            delta = td_target - v_s
            delta = delta.detach().numpy()  # (T, 1)

            # GAE(lambda) for advantage estimation
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)  # (T, 1)

            pi_logits = self.pi(s, first_hidden)[0].view(-1, dim_act)  # (T, 1, dim_a)
            pi = torch.softmax(pi_logits, dim=1)  # (T, dim_a)
            pi_a = pi.gather(1, a)  # (T, 1)
            ratio = pi_a / pi_k_a
            # ratio = torch.exp(torch.log(pi_a) - log_prob_a)  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss_pi = -torch.min(surr1, surr2)
            loss_v = F.smooth_l1_loss(v_s, td_target.detach())
            loss = loss_pi + loss_v

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

            kl_div_history.append(F.kl_div(F.log_softmax(pi_k_logits, dim=1),
                                           F.log_softmax(pi_logits, dim=1),
                                           log_target=True, reduction='batchmean').item())
            loss_pi_history.append(loss_pi.mean().item())
            loss_v_history.append(loss_v.mean().item())

        logs = {'loss_pi': np.mean(loss_pi_history),
                'loss_v': np.mean(loss_v_history)}
        for k, kl_div in enumerate(kl_div_history):
            logs[f'kl_div_{k}'] = kl_div
        return logs
