import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

device = "cuda" if torch.cuda.is_available() else "cpu"


class DQNAgent:

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            network,
            replay_method,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            double_dqn: bool = True,
            is_noisy: bool = True,
            is_categorical: bool = True,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,

            gamma: float = 0.99,

            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,

            n_step: int = 3,
            memory_n=None,

            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            PER: bool = False,
    ):

        self.n_states = n_states
        self.n_actions = n_actions
        self.memory = replay_method
        self.memory_counter = 0
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.is_noisy = is_noisy
        self.is_categorical = is_categorical
        self.n_step = n_step
        self.beta = beta
        self.prior_eps = prior_eps
        self.PER = PER
        self.loss_func = torch.nn.MSELoss()
        self.target_replace_iter = 100

        self.learn_step_counter = 0

        self.memory_size = memory_size

        self.use_n_step = True if n_step > 1 else False
        if (self.use_n_step):
            self.memory_n = memory_n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.dqn = network.to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08)

        self.transition = list()

        self.is_test = False

    def select_action(self, state: np.ndarray, list) -> np.ndarray:
        """Select an action from the input state."""

        if self.epsilon > np.random.random() and self.is_noisy == False:
            selected_action = np.random.choice(list)
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def store_transition(self, paper_state, action, reward, paper_state_, done) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.float64, bool]:

        if not self.is_test:

            self.transition = [paper_state, action, reward, paper_state_, done]

            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)

            else:
                one_step_transition = self.transition

        return paper_state, action, reward, paper_state_, done

    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch()

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        if (self.PER == True):
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)

            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = torch.mean(elementwise_loss)

        indices = samples["indices"]

        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            if (self.PER == True):
                weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
                loss = torch.mean(elementwise_loss * weights)
            else:
                loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        if (self.PER == True):
            loss_for_prior = elementwise_loss.detach()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        if (self.is_noisy == True):
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss

    def train(self, num_frames: int, paper_state, action, score, paper_state_, done, partition_list):
        """Train the agent."""
        self.is_test = False

        state = paper_state
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state, partition_list)
            state, action, reward, next_state, done = self.store_transition(paper_state, action, score, paper_state_,
                                                                            done)
            state = next_state
            score += reward

            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if done:
                state = paper_state
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

    def test(self, paper_state, action, reward, paper_state_, done, partition_list):
        """Test the agent."""
        self.is_test = True
        state = paper_state
        done = False
        score = 0

        while not done:
            action = self.select_action(paper_state, partition_list)
            state, action, next_state, reward, done = self.store_transition(paper_state, action, paper_state_, reward,
                                                                            done)

            state = next_state
            score += reward

        print("score while testing: ", score)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        if (self.is_categorical == True):
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            with torch.no_grad():
                if self.double_dqn == True:
                    next_action = self.dqn(next_state).argmax(1)
                else:
                    next_action = self.dqn_target(next_state).argmax(1)
                next_dist = self.dqn_target.dist(next_state)
                next_dist = next_dist[range(self.batch_size), next_action]

                t_z = reward + (1 - done) * self.gamma * self.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (
                    torch.linspace(
                        0, (self.batch_size - 1) * self.atom_size, self.batch_size
                    ).long()
                        .unsqueeze(1)
                        .expand(self.batch_size, self.atom_size)
                        .to(self.device)
                )

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

            dist = self.dqn.dist(state)
            log_p = torch.log(dist[range(self.batch_size), action])
            elementwise_loss = -(proj_dist * log_p).sum(1)
            return elementwise_loss
        else:

            curr_q_value = self.dqn(state).gather(1, action)

            if self.double_dqn == True:
                next_q_value = self.dqn_target(next_state).gather(
                    1, self.dqn(next_state).argmax(dim=1, keepdim=True)
                ).detach()
            else:
                next_q_value = self.dqn_target(
                    next_state
                ).max(dim=1, keepdim=True)[0].detach()

            mask = 1 - done
            target = (reward + self.gamma * next_q_value * mask).to(self.device)

            elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
            return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local.两种网络的更新"""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_states:])), self.q_eval(
            torch.Tensor(batch_memory[:, :self.n_states]))
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_states].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_states + 1])

        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        loss = torch.nn.MSELoss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss)

        self.epsilon = self.epsilon + self.epsilon_decay if self.epsilon < self.max_epsilon else self.max_epsilon
        self.learn_step_counter += 1
