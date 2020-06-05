
import argparse
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Playing options')

    parser.add_argument('--scenario', type=str, default='defend_the_center', metavar='S', help="scenario to use, either basic or deadly_corridor")
    parser.add_argument('--model_type', type=str, default='DQN', metavar='M', help="Model: DQN or DDDQN")
    parser.add_argument('--window', type=int, default=0, metavar='WIN', help="0: don't render screen | 1: render screen")
    parser.add_argument('--weights', type=str, metavar='S', help="Path to the weights we want to load")
    parser.add_argument('--total_episodes', type=int, default=100, metavar='EPOCHS', help="Number of training episodes")
    parser.add_argument('--frame_skip', type=int, default=2, metavar='FS', help="the number of frames to repeat the action on")
    parser.add_argument('--stack_size', type=int, default=4, metavar='FS', help="the number of frames to repeat the action on")

    args = parser.parse_args()

    test_environment(weights='models/best/DDDQN_80.pth', model_type='DDDQN', scenario=args.scenario, window=0,
        total_episodes = args.total_episodes, frame_skip=args.frame_skip, stack_size=args.stack_size)

    # test_a2c(weights='models/defend_the_center/DQN_90.pth', scenario=args.scenario, window=True,
    #     total_episodes=5, frame_skip=4, stack_size=4, resolution=(120, 160))