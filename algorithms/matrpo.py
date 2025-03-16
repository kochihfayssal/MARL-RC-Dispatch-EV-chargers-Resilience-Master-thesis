import torch
import os
import torch.functional as F
from networks import TRPOActor
from networks import TRPOCritic
from torch.distributions import Categorical
import numpy as np

class MATRPO:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        if args.last_action:
            self.actor_input_shape += self.n_actions
        if args.reuse_network:
            self.actor_input_shape += self.n_agents
        self.args = args
        self.device = torch.device("cuda")

        self.policy_rnn = TRPOActor(self.actor_input_shape, args)
        self.eval_critic = TRPOCritic(critic_input_shape, self.args)
        # self.target_critic = PPOCritic(critic_input_shape, self.args)

        if self.args.use_gpu:
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            # self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + "trial_3"
        
        # self.actor_parameters = list(self.policy_rnn.parameters()) 
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            # self.actor_optimizer = torch.optim.RMSprop(self.actor_parameters, lr=args.learning_rate)
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.learning_rate)
        elif args.optimizer == "Adam":
            # self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=args.learning_rate)
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=args.learning_rate)
        
        self.policy_hidden = None
        self.eval_critic_hidden = None
        # self.target_critic_hidden = None

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape
        # obs
        # input_shape += self.obs_shape
        # agent_id
        input_shape += self.n_agents

        # input_shape += self.n_actions * self.n_agents * 2  # 54
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
        if self.args.use_gpu:
            u = u.cuda()
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            s = s.cuda()
        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        for epoch in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)
            self.update_TRPO(batch, max_episode_len, u, mask, r, terminated)

    def compute_returns_advantages(self, batch, rewards, terminates, mask, max_episode_length):
        values, _ = self._get_values(batch, max_episode_length)
        values = values.detach().squeeze(dim=-1)
        returns = torch.zeros_like(rewards)
        deltas = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        prev_return = 0.0
        prev_value = 0.0
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
    
    def update_TRPO(self, batch, max_episode_length, actions, mask, r, terminated):
        returns, advantages = self.compute_returns_advantages(batch, r, terminated, mask, max_episode_length)
        values, _ = self._get_values(batch, max_episode_length)
        values = values.squeeze(dim=-1)
        squared_error = 0.5 * (values - returns)**2
        loss = (mask * squared_error).sum() / mask.sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.max_grad_norm)
        self.critic_optimizer.step()

        old_policy_net = self._clone_policy(self.policy_rnn)
        old_policy_net.cuda()
        loss = self.surrogate(self.policy_rnn, old_policy_net, batch, max_episode_length, actions, mask, advantages)
        loss_grad = torch.autograd.grad(loss, self.policy_rnn.parameters(), allow_unused=True)
        loss_grad_vector = self.flat_grad(loss_grad)
        # loss_grad_vector = torch.cat([grad.view(-1) for grad in loss_grad])
        step_dir = self.conjugate_gradient(self.policy_rnn, batch, max_episode_length, -loss_grad_vector)
        hvp = self.hessian_vector_product(self.policy_rnn, old_policy_net, batch, max_episode_length, step_dir)
        shs = 0.5 * (step_dir * hvp).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.args.max_kl)[0] #modif
        full_step = step_size * step_dir
        # step_size = torch.sqrt(2 * self.args.max_kl / (shs + 1e-8))
        expected_improve = (loss_grad_vector * full_step).sum(0, keepdim=True)
        params = self._get_flat_params(self.policy_rnn)
        flag = False
        fraction = 1
        for i in range(self.args.ls_step): 
            # Apply the update to the policy network
            new_params = params + fraction * full_step
            self._set_flat_params(self.policy_rnn, new_params)
            new_loss = self.surrogate(self.policy_rnn, old_policy_net, batch, max_episode_length, actions, mask, advantages)
            loss_improve = new_loss - loss
            kl = self.kl_divergence(self.policy_rnn, old_policy_net, batch, max_episode_length)
            if kl < self.args.max_kl and (loss_improve / expected_improve) > self.args.accept_ratio and loss_improve.item()>0:
                flag = True
                break
            expected_improve *= 0.5
            fraction *= 0.5
        if not flag:
            params = self._get_flat_params(old_policy_net)
            self._set_flat_params(self.policy_rnn, params)

    def surrogate(self, policy_net, old_policy_net, batch, max_episode_length, actions, mask, advantages):
        new_logits = self._get_action_prob(policy_net, batch, max_episode_length)
        old_logits = self._get_action_prob(old_policy_net, batch, max_episode_length)
        new_dist = Categorical(new_logits)
        old_dist = Categorical(old_logits)
        new_log_probs = new_dist.log_prob(actions.squeeze(dim=-1))
        old_log_probs = old_dist.log_prob(actions.squeeze(dim=-1))
        new_log_probs[mask == 0] = 0.0
        old_log_probs[mask == 0] = 0.0
        ratio = torch.exp(new_log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def conjugate_gradient(self, policy_net, batch, max_episode_length, b, nsteps=10, residual_tol=1e-10):
        """ Conjugate gradient to solve Ax = b where A is the Hessian of KL divergence. """
        x = torch.zeros_like(b).to(device=self.device)
        r = b.clone()
        p = b.clone()
        r_dot_r = torch.dot(r, r)

        for i in range(nsteps):
            hvp = self.hessian_vector_product(policy_net, policy_net, batch, max_episode_length, p)
            alpha = r_dot_r / (torch.dot(p, hvp)) #modif
            x += alpha * p
            r -= alpha * hvp
            new_r_dot_r = torch.dot(r, r)
            beta = new_r_dot_r / (r_dot_r) # modif : remove +1e-8
            p = r + beta * p
            r_dot_r = new_r_dot_r

            if r_dot_r < residual_tol:
                break

        return x

    def hessian_vector_product(self, policy_net, old_policy_net, batch, max_episode_length, v):
        """ Compute the Hessian-vector product of the KL divergence. """
        v.detach()
        kl = self.kl_divergence(policy_net, old_policy_net, batch, max_episode_length)
        kl_grad = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True, allow_unused=True)
        kl_grad_vector = self.flat_grad(kl_grad)
        # kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])

        kl_v = (kl_grad_vector * v).sum()
        kl_hessian = torch.autograd.grad(kl_v, policy_net.parameters(), allow_unused=True)
        hvp = self.flat_hessian(kl_hessian)
        # hvp = torch.cat([h.view(-1) for h in kl_hessian])

        return hvp + self.args.damping * v  # Add damping to improve numerical stability
    
    def kl_divergence(self, policy_net, old_policy_net, batch, max_episode_length):
        """ Compute the KL divergence between old and new policies. """
        new_logits = self._get_action_prob(policy_net, batch, max_episode_length)
        old_logits = self._get_action_prob(old_policy_net, batch, max_episode_length)

        new_dist = Categorical(new_logits)
        old_dist = Categorical(old_logits)

        kl_div = torch.distributions.kl.kl_divergence(new_dist, old_dist)
        return kl_div.mean()

    def _clone_policy(self, policy_net):
        cloned_net = type(policy_net)(self.actor_input_shape, self.args)
        cloned_net.load_state_dict(policy_net.state_dict())
        return cloned_net
    
    def _get_flat_params(self, model):
        return torch.cat([param.view(-1) for param in model.parameters()])
    
    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten
    
    def _set_flat_params(self, model, flat_params):
        start = 0
        for param in model.parameters():
            param_len = param.numel()
            param.data.copy_(flat_params[start:start+param_len].view_as(param))
            start += param_len
    
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
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                # inputs_next = inputs_next.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()
                # self.target_critic_hidden = self.target_critic_hidden.cuda()

            v_eval , self.eval_critic_hidden = self.eval_critic(inputs, self.eval_critic_hidden) 
            self.eval_critic_hidden = self.eval_critic_hidden.detach()
            # v_target, self.target_critic_hidden = self.eval_critic(inputs_next, self.target_critic_hidden)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            # v_target = v_target.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
            # v_targets.append(v_target)

        v_evals = torch.stack(v_evals, dim=1)
        # v_targets = torch.stack(v_targets, dim=1)
        return v_evals, v_targets
    
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
    
    def _get_action_prob(self, policy, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.policy_hidden = self.policy_hidden.cuda()
            outputs, self.policy_hidden = self.policy_rnn(inputs, self.policy_hidden)
            self.policy_hidden = self.policy_hidden.detach()
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cuda()
        action_prob = action_prob + 1e-10
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
    
    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.policy_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')


