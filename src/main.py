# System Imports
import __main__
import os
import argparse

# Local Relative Imports
from config.definitions import *
from ann.model import train_narx, train_noe, eval_ann
from rl.manager import RLManager
# from gp.model import PendulumGPManager

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
parser_rl.add_argument('--agent', type=str, choices=['q_learn', 'dqn', 'dqn_built', 'actor_critic'], default='q_learn',
                           required=False, help='Choose algorithm to determine the agent model')
parser_rl.add_argument('--env', type=str, choices=['unbalanced_disk', 'pendulum'], default='unbalanced_disk',
                           required=False, help='Choose environment to run')

def __main__():
    # Compile Arguments
    args = parser.parse_args()
    # print(parser.print_help())
    # print(args)

    if args.train:
        if args.method == 'gp':
            # PendulumGPManager(sparse=args.sparse, num_inducing_points=args.inducing)
            pass
        elif args.method == 'ann':
            if args.model_arch == 'narx':
                train_narx()
            elif args.model_arch == 'noe': 
                train_noe()
            print("Training of the model has been completed")
        elif args.method == 'rl':
            rlmanager = RLManager(method=args.agent, 
                                  env=args.env)
            rlmanager.train()
    elif args.test:
        if args.method == 'gp':
            # PendulumGPManager(sparse=args.sparse, num_inducing_points=args.inducing)
            pass
        elif args.method == 'ann':
            if eval_ann(args.mode_path, args.model_arch):
                print("Evaluation of the model has been completed")
            else:
                print("Model file does not exists")
        elif args.method == 'rl':
            rlmanager = RLManager(method=args.agent, 
                                  env=args.env)
            rlmanager.simulate()

if __name__ == '__main__':
    __main__()
