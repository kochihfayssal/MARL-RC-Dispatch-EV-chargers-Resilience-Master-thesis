import torch
import torch.nn as nn
import os
import torch.functional as F
from networks import DDPGActor
from networks import DDPGCritic
from torch.distributions import Categorical

class MADDPG:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        # if args.last_action:
        #     actor_input_shape += self.n_actions
        # if args.reuse_network:
        #     actor_input_shape += self.n_agents
        self.args = args # tau + batch_size + 

        self.actor_rnn = DDPGActor(actor_input_shape, args)
        self.critic_rnn = DDPGCritic(critic_input_shape, self.args)
        self.target_critic_rnn = DDPGCritic(critic_input_shape, self.args)
        self.target_critic_rnn.load_state_dict(self.critic_rnn.state_dict())

        if self.args.use_gpu:
            self.actor_rnn.cuda()
            self.critic_rnn.cuda()
            self.target_critic_rnn.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + "trial_2"

        if args.optimizer == "RMS":
            self.actor_optimizer = torch.optim.RMSprop(list(self.actor_rnn.parameters()), lr=args.learning_rate)
            self.critic_optimizer = torch.optim.RMSprop(list(self.critic_rnn.parameters()), lr=args.learning_rate)
        elif args.optimizer == "Adam":
            self.actor_optimizer = torch.optim.Adam(list(self.actor_rnn.parameters()), lr=args.learning_rate)
            self.critic_optimizer = torch.optim.Adam(list(self.critic_rnn.parameters()), lr=args.learning_rate)

        self.actor_hidden = None
        self.critic_hidden = None
        self.target_critic_hidden = None

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape
        # obs
        # input_shape += self.obs_shape
        # agent_id
        # input_shape += self.n_agents

        # input_shape += self.n_actions * self.n_agents * 2  # 54
        return input_shape

    def soft_update(self, target_net, source_net):
        """Soft update target network parameters."""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def learn(self, batch, max_episode_len, train_step,time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated, s, s_next = batch['u'], batch['r'],  batch['avail_u'], batch['terminated'], batch['s'], batch['s_next']

        mask = (1 - batch["padded"].float())

        if self.args.use_gpu:
            u = u.cuda()
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            s = s.cuda()
            s_next = s_next.cuda()
        
        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        for _ in range(self.args.ppo_n_epochs):
            _, next_q_values = self._get_values(batch, max_episode_len)
            next_q_values = next_q_values.max(-1)[0]
            target_q_values = torch.zeros_like(next_q_values)
            for transition_idx in range(max_episode_len):
                target_q_values[:, transition_idx] = r[:, transition_idx] + (
                    1 - terminated[:, transition_idx]) * self.args.gamma * next_q_values[:, transition_idx] * mask[:, transition_idx]
            current_q_values, _ = self._get_values(batch, max_episode_len)
            current_q_values = current_q_values.gather(1, u).squeeze(dim=-1)
            critic_loss = nn.MSELoss()(current_q_values * mask, target_q_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            critic_values, _ = self._get_values(batch, max_episode_len)
            actor_loss = critic_values.gather(1, self._get_action_prob(batch, max_episode_len).argmax(dim=-1, keepdim=True)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.target_critic_rnn, self.critic_rnn)

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        # u_onehot = batch['u_onehot'][:, transition_idx]
        # if transition_idx != max_episode_len - 1:
        #     u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        # else:
        #     u_onehot_next = torch.zeros(*u_onehot.shape)
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        # u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        # u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        #
        # if transition_idx == 0:
        #     u_onehot_last = torch.zeros_like(u_onehot)
        # else:
        #     u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
        #     u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        inputs, inputs_next = [], []

        inputs.append(s)
        inputs_next.append(s_next)

        # inputs.append(obs)
        # inputs_next.append(obs_next)

        # inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next
    
    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions, avail_next_actions = batch['avail_u'], batch['avail_u_next']
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.critic_hidden = self.critic_hidden.cuda()
                self.target_critic_hidden = self.target_critic_hidden.cuda()

            v_eval, self.critic_hidden = self.critic_rnn(inputs, self.critic_hidden)
            self.critic_hidden = self.critic_hidden.detach()
            v_target, self.target_critic_hidden = self.target_critic_rnn(inputs_next, self.target_critic_hidden)
            self.target_critic_hidden = self.target_critic_hidden.detach()
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            v_target = v_target.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
            v_targets.append(v_target)

        v_evals = torch.stack(v_evals, dim=1)
        v_targets = torch.stack(v_targets, dim=1)
        v_evals[avail_actions == 0.0] = 0.0
        v_targets[avail_next_actions == 0.0] = 0.0
        return v_evals, v_targets
    
    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        # if self.args.last_action:
        #     if transition_idx == 0:
        #         inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        #     else:
        #         inputs.append(u_onehot[:, transition_idx - 1])
        # if self.args.reuse_network:
        #     inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_action_prob(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.actor_hidden = self.actor_hidden.cuda()
            outputs, self.actor_hidden = self.actor_rnn(inputs, self.actor_hidden)
            self.actor_hidden = self.actor_hidden.detach()
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()
        # action_prob = action_prob + 1e-10

        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        
        action_prob = action_prob + 1e-10

        if self.args.use_gpu:
            action_prob = action_prob.cuda()
        return action_prob
    
    def init_hidden(self, episode_num):
        self.actor_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.critic_rnn.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.actor_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')


