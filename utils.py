from vizdoom import *
from collections import deque
import torch
import numpy as np
import random
import skimage
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


def stack_frames(stacked_frames, state, is_new_episode, maxlen=4, resize=(60, 80)):

    frame = screen_process(state, size=resize)
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

    return stacked_state, stacked_frames


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, device, actions):

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
    print(scenario)

    game.load_config("ViZDoom/scenarios/" + scenario + ".cfg")
    game.set_doom_scenario_path("ViZDoom/scenarios/" + scenario + ".wad")
    game.set_window_visible(window)
    game.init()

    if scenario == 'deadly_corridor':
        n_actions = 6
        actions = np.identity(6,dtype=int).tolist()
    else:
        n_actions = game.get_available_buttons_size()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        actions = [left, right, shoot]

    # actions = [list(a) for a in it.product([0, 1], repeat=n_actions)]

    return game, actions

def screen_process(screen, size=(60, 80)):

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

    game, actions = create_environment(scenario=scenario, window=window)
    rewards = []

    n_actions = len(actions)

    if model_type == 'DQN':
        model = DQN(
            stack_size=stack_size,
            n_actions=n_actions)
    else:
        model = DDDQN(
            stack_size=stack_size,
            n_actions=n_actions)

    state_dict = torch.load(weights, map_location=torch.device('cpu'))
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

        total_reward = game.get_total_reward()
        print ("Episode %d, Total reward: %.2f" %(i, total_reward))
        rewards.append(total_reward)
        time.sleep(0.1)

    game.close()

    print('Result (kill count)')
    print('mean', np.mean(rewards))
    print('std', np.std(rewards))
    print('max', np.max(rewards))
    print('min', np.min(rewards))



def test_a2c(weights, scenario='defend_the_center', window=False, total_episodes=5, frame_skip=2,
    stack_size=4, resolution=(120, 160)):

    game, actions = create_environment(scenario=scenario, window=window)

    n_actions = len(actions)

    agent = A2CAgent(actions, scenario, stack_size, resolution)

    state_dict = torch.load(weights, map_location=torch.device('cpu'))
    agent.actor.load_state_dict(state_dict)

    for i in range(total_episodes):
        game.new_episode()
        is_finished = game.is_episode_finished()
        state = get_state(game)
        state, stacked_frames = stack_frames(None, state, True, stack_size, resolution)
        while not is_finished:
            policy = agent.actor.forward(state)
            # print(policy.probs)
            action_id = torch.argmax(policy.probs)
            action = actions[action_id]
            # print(action)
            reward = game.make_action(action, frame_skip)
            is_finished = game.is_episode_finished()
            if not is_finished:
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, False, stack_size, resolution)

            time.sleep(0.02)

        print ("Total reward:", game.get_total_reward())
        time.sleep(0.1)

    game.close()