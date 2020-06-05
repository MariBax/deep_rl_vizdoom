from models import Actor, Critic
from memory import ReplayMemory
import torch
import torch.nn as nn
import numpy as np
from utils import *

from vizdoom import *
from collections import deque
import torch
from torch.autograd import Variable
import numpy as np
import random
import skimage
import skimage.transform
from models import *
import time
import itertools as it
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from a2c_agent import *

def setup_experiment(title, logdir):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    return writer, experiment_name


def stack_frames(stacked_frames, state, is_new_episode, maxlen=4, resize=(120, 160)):
    """
    stacked_frames : collections.deque object of maximum length maxlen.
    state          : the return of get_state() function.
    is_new_episode : boolean, if it's a new episode, we stack the same initial state maxlen times.
    maxlen         : Int, maximum length of stacked_frames (default=4)
    resize         : tuple, shape of the resized frame (default=(120,160))

    Returns
    --------------
    stacked_state  : 4-D Tensor, same information as stacked_frames but in tensor. This represents a state.
    stacked_frames : the updated stacked_frames deque.
    """

    # Preprocess frame
    frame = screen_process(state)
    frame = torch.tensor(frame, dtype=torch.float)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([frame[None] for i in range(maxlen)], maxlen=maxlen) # We add a dimension for the batch

        # Stack the frames
        stacked_state = torch.cat(tuple(stacked_frames), dim=0).unsqueeze(0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame[None]) # We add a dimension for the batch
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.cat(tuple(stacked_frames), dim=0).unsqueeze(0)

    return Variable(stacked_state), stacked_frames


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, device, actions):
    """
    explore_start    : Float, the initial exploration probability.
    explore_stop     : Float, the last exploration probability.
    decay_rate       : Float, the rate at which the exploration probability decays.
    state            : 4D-tensor (batch, motion, image)
    model            : DQNetwork
    actions          : List, the one-hot encoded possible actions.

    Returns
    -------------
    action              : np.array of shape (number_actions,), the action chosen by the greedy policy.
    explore_probability : Float, the exploration probability.
    """

    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(actions)

    else:
        Qs = model.forward(state.to(device))
        action = actions[int(torch.max(Qs, 1)[1][0])]

    return action, explore_probability


def create_environment(scenario='basic', window=False):
    """
    Description
    ---------------
    Creates VizDoom game instance along with some predefined possible actions.

    Parameters
    ---------------
    scenario : String, either 'basic' or 'deadly_corridor' or 'defend_the_center', the Doom scenario to use (default='basic')
    window   : Boolean, whether to render the window of the game or not (default=False)

    Returns
    ---------------
    game             : VizDoom game instance.
    actions : List, the one-hot encoded possible actions.
    """

    game = DoomGame()


    # Load the correct configuration

    game.load_config("ViZDoom/scenarios/defend_the_center.cfg")
    game.set_doom_scenario_path("ViZDoom/scenarios/defend_the_center.wad")
    game.set_window_visible(window)
    game.init()

    n_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n_actions)]
    # left = [1, 0, 0]
    # right = [0, 1, 0]
    # shoot = [0, 0, 1]
    # actions = [left, right, shoot]

    return game, actions

def screen_process(screen, size=(120, 160)):

    screen = skimage.transform.resize(screen, size)
    screen = skimage.color.rgb2gray(screen)

    return screen

def get_state(game):
    """
    Description
    --------------
    Get the current state from the game.

    Parameters
    --------------
    game : VizDoom game instance.

    Returns
    --------------
    state : 4-D Tensor, we add the temporal dimension.
    """

    state = game.get_state().screen_buffer # shape = (3, 480, 640)
    return np.transpose(state, [1, 2, 0])
    # return state[:, :, None] # shape = (3, 480, 1, 640)


def test_environment(weights, scenario='defend_the_center', model_type='DQN', window=False, total_episodes=3, frame_skip=2,
    stack_size=4):
    """
    weights        : String, path to .pth file containing the weights of the network we want to test.
    scenario       : String, either 'basic' or 'deadly_corridor', the Doom scenario to use (default='basic')
    window         : Boolean, whether to render the window of the game or not (default=False)
    total_episodes : Int, the number of testing episodes (default=100)
    enhance        : String, 'none' or 'dueling' (default='none')
    frame_skip     : Int, the number of frames to repeat the action on (default=2)

    Returns
    ---------------
    game             : VizDoom game instance.
    actions : List, the one-hot encoded possible actions.
    """

    game, actions = create_environment(scenario=scenario, window=window)

    n_actions = len(actions)

    if model_type == 'DQN':
        model = DQN(
            stack_size=stack_size,
            n_actions=n_actions)
    else:
        model = DDDQN(
            stack_size=stack_size,
            n_actions=n_actions)

    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)

    for i in range(total_episodes):
        game.new_episode()
        is_finished = game.is_episode_finished()
        state = get_state(game)
        state, stacked_frames = stack_frames(None, state, True, stack_size)
        while not is_finished:
            q = model.forward(state)

            # action = random.choice(actions)

            action = actions[int(torch.max(q, 1)[1][0])]
            reward = game.make_action(action, frame_skip)
            is_finished = game.is_episode_finished()
            if not is_finished:
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, False, stack_size)

            time.sleep(0.02)

        print ("Total reward:", game.get_total_reward())
        time.sleep(0.1)

    game.close()



def test_a2c(weights, scenario='defend_the_center', window=False, total_episodes=5, frame_skip=2,
    stack_size=4, resolution=(120, 160)):

    game, actions = create_environment(scenario=scenario, window=window)

    n_actions = len(actions)

    agent = A2CAgent(actions, stack_size, resolution)

    state_dict = torch.load(weights)
    agent.actor.load_state_dict(state_dict)

    for i in range(total_episodes):
        game.new_episode()
        is_finished = game.is_episode_finished()
        state = get_state(game)
        state, stacked_frames = stack_frames(None, state, True, stack_size)
        while not is_finished:
            policy = agent.actor.forward(state)
            action_id = torch.argmax(policy.probs)
            action = actions[action_id]
            reward = game.make_action(action, frame_skip)
            is_finished = game.is_episode_finished()
            if not is_finished:
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, False, stack_size)

            time.sleep(0.02)

        print ("Total reward:", game.get_total_reward())
        time.sleep(0.1)

    game.close()


class A2CAgent:
    def __init__(self, actions, scenario, stack_size, resolution):
        # possible actions
        self.actions = actions
        self.scenario = scenario

        # sizes
        self.n_actions = len(actions)
        self.stack_size = stack_size
        self.resolution = resolution

        # memory for one episode
        self.rewards = []
        self.log_probs = []
        self.values = []

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # networks
        self.actor = Actor(self.n_actions, stack_size).to(self.device)
        self.critic = Critic(1, stack_size).to(self.device)

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


    def train(self, environment, writer=None, total_episodes=20, frame_skip=4,
        actor_lr=1e-4, critic_lr=1e-4, freq=3):

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        torch.set_default_dtype(torch.float)

        for episode in range(total_episodes):

            # init new episode
            environment.game.new_episode()
            state = environment.get_state()
            obs_cur, obs_prev = environment.init_observations()

            # prepare stack of frames
            state, stacked_frames = stack_frames(None, state, True, self.stack_size, self.resolution)
            state = Variable(state, requires_grad=True)

            # check episode is finished
            is_finished = environment.is_episode_finished()

            while not is_finished:
                # sample action from stochastic softmax policy
                policy, value = self.actor(state.to(self.device)), self.critic(state.to(self.device))

                action_id = policy.sample()
                log_prob = policy.log_prob(action_id).unsqueeze(0)
                action = self.actions[action_id]

                # make action and get reward
                reward = environment.make_action(action, frame_skip)
                obs_cur = environment.get_observation_cur()
                reward = self._reshape_reward(reward, obs_cur, obs_prev)
                obs_prev = obs_cur.copy()

                # fill memory
                self.log_probs.append(log_prob)
                self.rewards.append(reward)
                self.values.append(value)

                # check episode is finished
                is_finished = environment.is_episode_finished()

                if not is_finished:
                    # get new state
                    next_state = environment.get_state()
                    state, stacked_frames = stack_frames(
                        stacked_frames, next_state, False, self.stack_size, self.resolution)
                    state = Variable(state, requires_grad=True)

                else:
                    # every episode agent learns
                    print('Episode finished, training...')
                    actor_loss, critic_loss = self.train_on_episode()
                    episode_reward = sum(self.rewards)
                    kill_count = obs_cur['kills']
                    print(
                        "Episode: %d, Total reward: %.2f, Kill Count: %.1f, Actor loss: %.4f, Critic loss: %.4f" % (
                            episode, episode_reward, kill_count, actor_loss, critic_loss))
                    self.log_probs, self.rewards, self.values = [], [], []

                    # save model
                    if (episode % freq) == 0:
                        model_file = 'models/' + environment.scenario + '/' + 'A2C' + '_' + str(episode) + '.pth'
                        torch.save(self.actor.state_dict(), model_file)
                        print('Saved model to ' + model_file)


    def discount_rewards(self, gamma=0.99):
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            if self.rewards[t] != 0:
                running_add = 0
            running_add = running_add * gamma + self.rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train_on_episode(self):

        discounted_rewards = torch.tensor(self.discount_rewards(), dtype=torch.float).reshape(-1, 1).to(self.device)
        log_probs = torch.cat(self.log_probs).to(self.device)
        values = torch.cat(self.values).to(self.device)

        # print(log_probs.requires_grad)
        # print(discounted_rewards.shape)
        # print(values.requires_grad)

        advantage = discounted_rewards - values

        # print(advantage.requires_grad)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.cpu().item(), critic_loss.cpu().item()


