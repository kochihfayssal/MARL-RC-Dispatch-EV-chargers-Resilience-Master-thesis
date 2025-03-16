import torch
import os
import torch.functional as F
from networks import A2CActor
from networks import A2CCritic
from torch.distributions import Categorical

class MAA2C:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        self.policy_rnn = A2CActor(actor_input_shape, args)
        self.eval_critic = A2CCritic(critic_input_shape, self.args)
        # self.target_critic = PPOCritic(critic_input_shape, self.args)

        if self.args.use_gpu:
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            # self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + "trial_1"

        self.ac_parameters = list(self.policy_rnn.parameters()) + list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.ac_optimizer = torch.optim.RMSprop(self.ac_parameters, lr=args.learning_rate)
        elif args.optimizer == "Adam":
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=args.learning_rate)

        self.policy_hidden = None
        self.eval_critic_hidden = None
        self.target_critic_hidden = None
    
    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape
        # obs
        input_shape += self.n_agents
        return input_shape
    
    def learn(self, batch, max_episode_len, train_step,time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated, s = batch['u'], batch['r'],  batch['avail_u'], batch['terminated'], batch['s']

        mask = (1 - batch["padded"].float())

    def compute_returns_advantages(self, batch, rewards, terminates, mask, max_episode_length):
        values, next_values = self._get_values(batch, max_episode_length)
        values = values.detach().squeeze(dim=-1)
        next_values = next_values.detach().squeeze(dim=-1)
        returns = torch.zeros_like(rewards)
        deltas = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        prev_return = next_values[:, -1]
        prev_value = next_values[:, -1]
        prev_advantage = 0.0
        for transition_idx in reversed(range(max_episode_length)):
            returns[:,transition_idx] = rewards[:,transition_idx] + self.args.gamma * prev_return * (1-terminates[:,transition_idx]) * mask[:, transition_idx]
            deltas[:,transition_idx] = rewards[:,transition_idx] + self.args.gamma * prev_value * (1-terminates[:,transition_idx]) * mask[:, transition_idx] \
                - values[:, transition_idx]
            advantages[:,transition_idx] = deltas[:,transition_idx] + self.args.gamma * self.args.gae_lambda * prev_advantage * (
                1-terminates[:,transition_idx]) * mask[:, transition_idx]
            prev_return = returns[:,transition_idx]
            prev_value = values[:,transition_idx]
            prev_advantage = advantages[:,transition_idx]
        advantages = (advantages - advantages.mean()) / ( advantages.std() + 1e-8)
        return returns, advantages
    
    def update_policy_and_value(self, batch, max_episode_length, actions, advantages, returns, mask):
        """ Update the policy and value networks using the A2C loss. """
        # Get policy logits and state values from networks
        logits = self._get_action_prob(batch, max_episode_length)
        values, _ = self._get_values(batch, max_episode_length)
        values = values.squeeze(dim=-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(dim=-1))
        entropy = dist.entropy()
        entropy[mask == 0] = 0.0
        policy_loss = -(log_probs * advantages.detach()).mean()
        squared_error = 0.5*(values - returns)**2
        value_loss = (mask * squared_error).sum() / mask.sum()
        entropy_loss = entropy.mean()
        loss = policy_loss + value_loss - self.args.ent_coef * entropy_loss
        self.ac_optimizer.zero_grad()
        loss.backward()
        self.ac_optimizer.step()
    
    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]

        inputs, inputs_next = [], []

        inputs.append(s)
        inputs_next.append(s_next)

        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next
    
    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        v_evals, v_next_evals = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()
                self.target_critic_hidden = self.target_critic_hidden.cuda()

            v_eval, self.eval_critic_hidden = self.eval_critic(inputs, self.eval_critic_hidden)
            v_target, self.target_critic_hidden = self.eval_critic(inputs_next, self.target_critic_hidden)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            v_target = v_target.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
            v_next_evals.append(v_target)

        v_evals = torch.stack(v_evals, dim=1)
        v_next_evals = torch.stack(v_next_evals, dim=1)
        return v_evals, v_next_evals
    
    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

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
                self.policy_hidden = self.policy_hidden.cuda()
            outputs, self.policy_hidden = self.policy_rnn(inputs, self.policy_hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cuda()
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        
        action_prob = action_prob + 1e-10

        if self.args.use_gpu:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        self.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.policy_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')