from vizdoom import *
from utils import *
import torch
import torch.nn as nn
from collections import namedtuple, deque
from memory import ReplayMemory, PrioritizedMemory
from models import DQN, DDDQN
import random
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_finished'))

class Agent:

    def __init__(self, actions, scenario='defend_the_center', memory_type='prioritized', capacity=100, stack_size=4,
                 resolution=(120, 160)):

        self.actions = actions
        self.scenario = scenario
        self.memory_type = memory_type

        if memory_type == 'uniform':
            self.memory = ReplayMemory(capacity)
        else:
            self.memory = PrioritizedMemory(capacity)

        self.stack_size = stack_size
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def append_sample(self, state, action, next_state, reward, is_finished):
        target = self.model(state.to(self.device)).data
        action_id = list(action).index(1)
        # print(action_id)
        old_val = target[0][action_id]
        target_val = self.target_model(next_state.to(self.device)).data
        if is_finished:
            target[0][action] = reward
        else:
            target[0][action] = reward + 0.99 * torch.max(target_val)

        error = abs(old_val - target[0][action_id])
        # print('error', error)

        if not torch.is_tensor(action):
            action = torch.tensor([action], dtype=torch.float)
        if not torch.is_tensor(reward):
            reward = torch.tensor([reward], dtype=torch.float)
        if not torch.is_tensor(is_finished):
            is_finished = torch.tensor([is_finished], dtype=torch.float)

        self.memory.push(error.cpu(), (state, action, next_state, reward, is_finished))

    def _reshape_reward(self, reward, obs_cur, obs_prev):
        """
        obs_cur  : dict containing current observation (kills, health and ammo).
        obs_prev : dict containing previous observation (kills, health and ammo).
        """
        if self.scenario == 'basic':
            return reward
        elif self.scenario == 'defend_the_center':
            if obs_cur['health'] < obs_prev['health']:
                health_decrease = -0.2
            else:
                health_decrease = 0

            if obs_cur['ammo'] < obs_prev['ammo'] and reward <= 0:
                miss_penalty = -0.2
            else: # miss
                miss_penalty = 0

            return reward + miss_penalty + health_decrease

        elif self.scenario == 'deadly_corridor':
            ammo_decrease = int(obs_cur['ammo'] < obs_prev['ammo'])
            health_decrease = int(obs_cur['health'] < obs_prev['health'])
            penalty = -0.5 * (ammo_decrease + health_decrease)
            kill_reward = (obs_prev['kills'] - obs_cur['kills']) * 40
            return reward / 2.0 - penalty + kill_reward


    def _pretrain(self, environment, iter_num=100, frame_skip=4):
        # init new episode
        environment.game.new_episode()
        obs_cur, obs_prev = environment.init_observations()

        # get state and generate stack of frames
        state = environment.get_state()
        state, stacked_frames = stack_frames(None, state, True, self.stack_size, self.resolution)
        # print(state.shape)

        # perform random actions and save transition to memory
        for i in range(iter_num):
            # select action
            action = random.choice(self.actions)

            # make action and update obserwations
            reward = environment.make_action(action, frame_skip)
            obs_cur = environment.get_observation_cur()
            reward = self._reshape_reward(reward, obs_cur, obs_prev)
            obs_prev = obs_cur.copy()

            # episode finished?
            is_finished = environment.is_episode_finished()

            if is_finished:
                next_state = np.zeros((480, 640, 3), dtype='uint8')
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resolution)

                self.append_sample(state, action, next_state, reward, is_finished)
                # self.memory.push(state, action, next_state, reward, is_finished)

                environment.game.new_episode()
                state = environment.get_state()

                state, stacked_frames = stack_frames(
                    stacked_frames, state, True, self.stack_size, self.resolution)

            else:
                next_state = environment.get_state()
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False, self.stack_size, self.resolution)
                self.append_sample(state, action, next_state, reward, is_finished)
                # self.memory.push(state, action, next_state, reward, is_finished)
                state = next_state


    def train(self, environment, model_type='DQN', writer=None, total_episodes=100, pretrain=100, batch_size=64,
        frame_skip=4, lr=1e-4, explore_start=1.0,
        explore_stop=0.01, decay_rate=0.001, gamma=0.99, freq=5):

        print('...Using device...', self.device)
        torch.set_default_dtype(torch.float)

        n_actions= len(self.actions)

        if model_type == 'DQN':
            self.model = DQN(
                stack_size=self.stack_size,
                n_actions=n_actions).to(self.device)
            self.target_model = DQN(
                stack_size=self.stack_size,
                n_actions=n_actions).to(self.device)
        else:
            self.model = DDDQN(
                stack_size=self.stack_size,
                n_actions=n_actions).to(self.device)
            self.target_model = DDDQN(
                stack_size=self.stack_size,
                n_actions=n_actions).to(self.device)

        self.target_model.eval()

        print('...Filling replay memory...')
        self._pretrain(environment, iter_num=pretrain)

        print('...Training loop...')
        max_tau = 150
        global_i = 0

        # act_to_ind = {tuple(action): i for i, action in enumerate(self.actions)}

        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        kill_count_ma = deque(maxlen=10)

        for episode in range(total_episodes):
            tau = 0
            decay_step = 0
            episode_rewards = []
            episode_loss = 0
            episode_len = 1

            environment.game.new_episode()
            obs_cur, obs_prev = environment.init_observations()

            state = environment.get_state()
            state, stacked_frames = stack_frames(None, state, True, self.stack_size, self.resolution)

            is_finished = environment.is_episode_finished()

            while not is_finished:
                tau += 1
                decay_step += 1

                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                    decay_step, state, self.model, self.device, self.actions)
                reward = environment.make_action(action, frame_skip)
                # print(reward)

                obs_cur = environment.get_observation_cur()
                reward = self._reshape_reward(reward, obs_cur, obs_prev)
                # print(reward)
                obs_prev = obs_cur.copy()

                is_finished = environment.is_episode_finished()

                episode_rewards.append(reward)

                if is_finished:
                    next_state = np.zeros((480, 640, 3), dtype='uint8')
                    next_state, stacked_frames = stack_frames(
                        stacked_frames, next_state, False, self.stack_size, self.resolution)
                    episode_reward = np.sum(episode_rewards)
                    kill_count = obs_cur['kills']
                    kill_count_ma.append(kill_count)
                    print(
                        "Episode: %d, Total reward: %.2f, Kill count: %.1f, Train loss: %.4f, Explore p: %.4f" % (
                            episode, episode_reward, np.mean(kill_count_ma), loss, explore_probability))
                    self.append_sample(state, action, next_state, reward, is_finished)
                    # self.memory.push(state, action, next_state, reward, is_finished)

                else:
                    next_state = environment.get_state()
                    next_state, stacked_frames = stack_frames(
                        stacked_frames, next_state, False, self.stack_size, self.resolution)
                    self.append_sample(state, action, next_state, reward, is_finished)
                    # self.memory.push(state, action, next_state, reward, is_finished)
                    state = next_state
                    episode_len += 1

                self.model.train()

                transitions, idxs, is_weights = self.memory.sample(batch_size)

                batch = Transition(*zip(*transitions))
                states_mb = torch.cat(batch.state).to(self.device)
                actions_mb = torch.cat(batch.action).to(self.device)
                rewards_mb = torch.cat(batch.reward).to(self.device)
                next_states_mb = torch.cat(batch.next_state).to(self.device)
                dones_mb = torch.cat(batch.is_finished).to(self.device)

                q_next_state = self.model.forward(next_states_mb)
                q_target_next_state = self.target_model(next_states_mb)
                q_state = self.model.forward(states_mb)

                targets_mb = rewards_mb + (gamma * (1 - dones_mb) * torch.max(q_target_next_state, 1)[0])
                # actions_ids = [self.actions.index(action.int().tolist()[0]) for action in batch.action]
                # q_values_for_actions = q_state.contiguous()[np.arange(q_state.shape[0]), actions_ids]

                output = (q_state * actions_mb).sum(1)
                errors = torch.abs(output - targets_mb).cpu().data.numpy()

                # update priority
                for i in range(batch_size):
                    idx = idxs[i]
                    self.memory.update(idx, errors[i])

                optimizer.zero_grad()

                # loss = criterion(q_values_for_actions, targets_mb)
                loss = (torch.tensor(is_weights, dtype=torch.float, device=self.device) * criterion(output, targets_mb.detach())).mean()
                # loss = criterion(output, targets_mb.detach())
                loss.backward()

                for p in self.model.parameters():
                    p.grad.data.clamp_(-1, 1)

                optimizer.step()

                episode_loss += loss.item()

                # dump train metrics to tensorboard
                if writer is not None:
                    writer.add_scalar("loss/train", loss.item(), global_i)
                    writer.add_scalar("reward/train", torch.mean(rewards_mb).item(), global_i)

                if tau > max_tau:
                    self.target_model.load_state_dict(self.model.state_dict())
                    print('Target model updated')
                    tau = 0

                global_i += 1

            # dump episode metrics to tensorboard
            if writer is not None:
                writer.add_scalar("loss_episode/train", episode_loss / episode_len, episode)
                writer.add_scalar("reward_episode/train", episode_reward, episode)
                writer.add_scalar("kill_count/train", kill_count, episode)

            if (episode % freq) == 0:
                model_file = 'models/' + self.scenario + '/' + model_type + '_' + str(episode) + '.pth'
                torch.save(self.model.state_dict(), model_file)
                print('\nSaved model to ' + model_file)

