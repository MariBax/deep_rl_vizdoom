from vizdoom import *
import numpy as np
import itertools as it

class DoomEnvironment:
    def __init__(self, scenario='defend_the_center', window=False):
        self.game = DoomGame()
        print(scenario)

        self.game.load_config("ViZDoom/scenarios/" + scenario + ".cfg")
        self.game.set_doom_scenario_path("ViZDoom/scenarios/" + scenario + ".wad")
        self.game.set_window_visible(window)
        self.game.init()

        self.scenario = scenario

        if self.scenario == 'deadly_corridor':
            self.n_actions = 6
            self.actions = np.identity(6,dtype=int).tolist()
        else:
            self.n_actions = self.game.get_available_buttons_size()
            left = [1, 0, 0]
            right = [0, 1, 0]
            shoot = [0, 0, 1]
            self.actions = [left, right, shoot]

        # self.actions = [list(a) for a in it.product([0, 1], repeat=self.n_actions)]

    def get_actions(self):
        return self.actions

    def init_observations(self):
        obs_cur = self.get_observation_cur()
        obs_prev = obs_cur.copy()
        return obs_cur, obs_prev

    def get_observation_cur(self):
        obs_cur = {}
        obs_cur['kills'] = self.game.get_game_variable(KILLCOUNT)
        obs_cur['health'] = self.game.get_game_variable(HEALTH)
        obs_cur['ammo'] = self.game.get_game_variable(AMMO2)
        return obs_cur

    def get_state(self):
        state = self.game.get_state().screen_buffer # shape = (3, 480, 640)
        return np.transpose(state, [1, 2, 0]) # shape = (480, 640, 3)

    def get_zero_state(self):
        return np.zeros((480, 640, 3), dtype='uint8')

    def make_action(self, action, frame_skip):
        return self.game.make_action(action, frame_skip)

    def is_episode_finished(self):
        return self.game.is_episode_finished()