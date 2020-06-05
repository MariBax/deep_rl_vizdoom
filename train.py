import argparse
from agent import *
# from a2c_agent import *
# from a2c import *
from environment import *
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train options')

    parser.add_argument('--scenario', type=str, default='defend_the_center', metavar='S', help="scenario to use, either basic or deadly_corridor")
    parser.add_argument('--model_type', type=str, default='DQN', metavar='M', help="Model: DQN or DDDQN")
    parser.add_argument('--window', type=int, default=0, metavar='WIN', help="0: don't render screen | 1: render screen")
    parser.add_argument('--resolution', type=tuple, default=(60, 80), metavar='RES', help="Size of the resolutiond frame")
    parser.add_argument('--stack_size', type=int, default=4, metavar='SS', help="Number of frames to stack to create motion")
    parser.add_argument('--explore_start', type=float, default=1., metavar='EI', help="Initial exploration probability")
    parser.add_argument('--explore_stop', type=float, default=0.01, metavar='EL', help="Final exploration probability")
    parser.add_argument('--decay_rate', type=float, default=1e-3, metavar='DR', help="Decay rate of exploration probability")
    parser.add_argument('--capacity', type=int, default=1000, metavar='MS', help="Size of the experience replay buffer")
    parser.add_argument('--memory_type', type=str, default='prioritized', metavar='S', help="uniform or prioritized")
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help="Batch size")
    parser.add_argument('--gamma', type=float, default=.99, metavar='GAMMA', help="Discounting rate")
    parser.add_argument('--total_episodes', type=int, default=100, metavar='EPOCHS', help="Number of training episodes")
    parser.add_argument('--pretrain', type=int, default=64, metavar='PRE', help="number of initial experiences to put in the replay buffer")
    parser.add_argument('--frame_skip', type=int, default=4, metavar='FS', help="the number of frames to repeat the action on")
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR', help="The learning rate")
    parser.add_argument('--freq', type=int, default=10, metavar='FQ', help="Number of episodes to save model weights")

    args = parser.parse_args()
    # game, possible_actions = create_environment(scenario=args.scenario, window=args.window)
    # writer, experiment_name = setup_experiment(title='DQN', logdir='logs_dqn')
    # agent = Agent(game, possible_actions, args.scenario, args.capacity, args.stack_size,
    #              args.batch_size, args.resolution)
    # agent.train('DQN', writer, args.total_episodes, args.pretrain, args.frame_skip, args.lr,
    #     args.explore_start, args.explore_stop, args.decay_rate, args.gamma, args.freq)

    # print('...Creating environment...')
    # environment = DoomEnvironment(args.scenario, args.window)
    # print('...Creating agent...')
    # agent = A2CAgent(environment.actions, args.scenario, args.stack_size, args.resolution)
    # print('...Training...')
    # agent.train(environment, total_episodes=150, frame_skip=4, freq=3)

    print('...Creating environment...')
    environment = DoomEnvironment(args.scenario, args.window)
    possible_actions = environment.get_actions()
    writer, experiment_name = setup_experiment(title='DDDQN', logdir='logs_dqn')

    print('...Creating agent...')
    agent = Agent(possible_actions, args.scenario, args.memory_type, args.capacity, args.stack_size, args.resolution)

    print('...Training...')
    agent.train(environment, 'DDDQN', writer, args.total_episodes, args.pretrain, args.batch_size,
        args.frame_skip, args.lr, args.explore_start, args.explore_stop,
        args.decay_rate, args.gamma, args.freq)