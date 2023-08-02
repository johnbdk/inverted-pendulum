# System Imports
import __main__
import os
import argparse

# Local Relative Imports
from config.definitions import *
from ann.model import train_narx, train_noe, eval_ann, grid_search
from rl.manager import RLManager
# from gp.model import PendulumGPManager
from config.rl import TRAIN_STEPS, TEST_STEPS

###########  P A R A M E T E R S  ###########

# PARENT PARSER FOR SOME SUBPARSERS TO INHERIT THE ARGUMENTS IN ORDER TO BE SHARED, THUS WE DONT NEED
# TO SPECIFY EACH SHARED ARGUMENTS TO EACH SEPERATE SUBPARSER
parent_parser = argparse.ArgumentParser(description = 'put some description here...', add_help=False)
parent_parser.add_argument('--train', action='store_true', default=False, help='Train the model')
parent_parser.add_argument('--test', action='store_true', default=False, help='Train the model')
parent_parser.add_argument('--verbose', help='Common top-level parameter',
                    action='store_true', required=False)
# parent_parser.add_argument("--action", action='store_true', help="action of the mode to be used")
# parent_parser.add_argument('--save', action='store_true', default=False, help='Save the model')

## General Arguments
parent_parser.add_argument('--model-file', type = str, default = os.path.join(MODELS_DIR, "model.pth"),
                           required=False, help = 'Specify the model path to be stored/loaded')
parent_parser.add_argument('--model-arch', type=str, choices=['narx', 'noe', 'ss'], default='narx',
                           required=False, help='Choose model architecture')

# Define Arguments
parser = argparse.ArgumentParser(parents=[parent_parser])
subparsers = parser.add_subparsers(title="actions", dest="method")
parser_ann = subparsers.add_parser("ann", parents=[parent_parser],
                                      description='The method parser', help='Method to be chosen (ANN or GP or RL)')
# parser_ann.add_argument('--model-arch', type=str, choices=['narx', 'noe', 'ss'], default='narx',
#                            required=False, help='Choose model architecture')

parser_gp = subparsers.add_parser("gp", parents=[parent_parser],
                                      description='The method parser', help='Method to be chosen (ANN or GP or RL)')
parser_gp.add_argument('--sparse', action='store_true', default=False, help="name of the method to be used")
parser_gp.add_argument('--inducing', type=int, default=0, help='Train the model')


parser_rl = subparsers.add_parser("rl", parents=[parent_parser],
                                      description='The method parser', help='Method to be chosen (ANN or GP or RL)')
parser_rl.add_argument('--agent', type=str, choices=['q_learn', 'dqn', 'a2c', 'a2c_built'], default='q_learn',
                           required=False, help='Choose algorithm to determine the agent model')
parser_rl.add_argument('--env', type=str, choices=['unbalanced_disk', 'pendulum'], default='unbalanced_disk',
                           required=False, help='Choose environment to run')
parser_rl.add_argument('--render', action='store_true', default=False, help="whether or not to render environment during training")

parser_rl.add_argument('--load', type=str, default='QLearning_Best_0', required=False, help='Path of model to load')

def __main__():
    # Compile Arguments
    args = parser.parse_args()

    # 1. SYSTEM IDENTIFICATION : Gaussian Process Task
    if args.method == 'gp':
        if args.train:
            # PendulumGPManager(sparse=args.sparse, num_inducing_points=args.inducing)
            pass
        elif args.test:
            # PendulumGPManager(sparse=args.sparse, num_inducing_points=args.inducing)
            pass

    # 2. SYSTEM IDENTIFICATION : Artificial Neural Network Task
    elif args.method == 'ann':
        if args.train:
            if args.model_arch == 'narx':
                train_narx(2,2,32)
            elif args.model_arch == 'narx_grid':
                train_narx(2,2,32)
                grid_search()
            elif args.model_arch == 'noe': 
                train_noe()
            print("Training of the model has been completed")
        elif args.test:
            if eval_ann(args.mode_path, args.model_arch):
                print("Evaluation of the model has been completed")
            else:
                print("Model file does not exists")

    # 3. POLICY LEARNING : Reinforcement Learning
    elif args.method == 'rl':
        # perform task
        if args.train:
            rlm = RLManager(env=args.env,
                            method=args.agent,
                            mode='train',
                            train_steps=TRAIN_STEPS)
            rlm.train(render=args.render)
        elif args.test:
            rlm = RLManager(env=args.env,
                            method=args.agent,
                            mode='test',
                            test_steps=TEST_STEPS,
                            model_path = args.load)
            rlm.simulate()

if __name__ == '__main__':
    __main__()
