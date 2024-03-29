# System Imports
import __main__
import os
import argparse

# Local Relative Imports
from config.definitions import *
from ann.model import train_narx, train_noe, eval_noe, eval_narx, simulation_narx, prediction_narx, prediction_noe, simulation_noe, train_narx_grid, eval_grid
from rl.manager import RLManager
from gp.manager import GPManager
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
parser_ann.add_argument('--grid_search', action='store_true', default=False, required=False, help="do a grid search for hyperparameters optimization")
parser_ann.add_argument('--grid_eval', action='store_true', default=False, required=False, help="do a grid evaluation for hyperparameters optimization")
parser_ann.add_argument('--pred_submission', action='store_true', default=False, required=False, help="do a prediction submission")
parser_ann.add_argument('--sim_submission', action='store_true', default=False, required=False, help="do a prediction submission")
# parser_ann.add_argument('--model-arch', type=str, choices=['narx', 'noe', 'ss'], default='narx',
#                            required=False, help='Choose model architecture')

parser_gp = subparsers.add_parser("gp", parents=[parent_parser],
                                      description='The method parser', help='Method to be chosen (ANN or GP or RL)')
parser_gp.add_argument('--nb', type=int, default=3, help='Number of past inputs of NARX')
parser_gp.add_argument('--na', type=int, default=3, help='Number of past outputs of NARX')
parser_gp.add_argument('--sparse', action='store_true', default=False, help="Method to be used (Sparse or Full Gaussian process)")
parser_gp.add_argument('--inducing', type=int, default=1, help='Number of inducing points (used in sparse Gaussian process)')
parser_gp.add_argument('--samples', type=int, default=-1, help='Number of samples to be used. Default -1 (this means full training data)')
parser_gp.add_argument('--fname', type=str, required=False, help='Name of model to load')
parser_gp.add_argument('--sim', action='store_true', default=False, required=False, help="whether or not to do simulation or prediction")
parser_gp.add_argument('--grid_search', action='store_true', default=False, required=False, help="do a grid search for hyperparameters optimization")
parser_gp.add_argument('--load_grid_search', action='store_true', default=False, required=False, help="do a grid search for hyperparameters optimization")
parser_gp.add_argument('--pred_submission', action='store_true', default=False, required=False, help="do a prediction submission")
parser_gp.add_argument('--sim_submission', action='store_true', default=False, required=False, help="do a prediction submission")


parser_rl = subparsers.add_parser("rl", parents=[parent_parser],
                                      description='The method parser', help='Method to be chosen (ANN or GP or RL)')
parser_rl.add_argument('--agent', type=str, choices=['q_learn', 'dqn', 'a2c', 'a2c_built'], default='q_learn',
                           required=False, help='Choose algorithm to determine the agent model')
parser_rl.add_argument('--env', type=str, choices=['unbalanced_disk', 'pendulum'], default='unbalanced_disk',
                           required=False, help='Choose environment to run')
parser_rl.add_argument('--multi_target', action='store_true', default=False, 
                       help='Choose environment with single or multiple targets (applies to the unbalanced disk env only!)')
parser_rl.add_argument('--render', action='store_true', default=False, help="whether or not to render environment during training")
parser_rl.add_argument('--load', type=str, default='QLearning_Best_0', required=False, help='Path of model to load')

def __main__():
    # Compile Arguments
    args = parser.parse_args()

    # 1. SYSTEM IDENTIFICATION : Gaussian Process Task
    if args.method == 'gp':
        gpm = GPManager(num_inputs=args.nb,
                        num_outputs=args.na,
                        sparse=args.sparse,
                        num_inducing=args.inducing,
                        num_samples=args.samples)
        if args.train:
            gpm.train()
        elif args.test:
            gpm.test(fname=args.fname, load=True, sim=args.sim)
        elif args.grid_search:
            gpm.grid_search()
        elif args.load_grid_search:
            gpm.load_grid_search_dict()
        elif args.pred_submission:
            gpm.test_prediction_submission(fname=args.fname)
        elif args.sim_submission:
            gpm.test_simulation_submission(fname=args.fname)

    # 2. SYSTEM IDENTIFICATION : Artificial Neural Network Task
    elif args.method == 'ann':
        if args.train:
            if args.model_arch == 'narx':
                train_narx(2, 2, 32)
            elif args.model_arch == 'noe': 
                train_noe()
            print("Training of the model has been completed")
        elif args.test:
            if args.model_arch == 'narx':
                eval_narx(2, 2)
                print("Evaluation of the NARX model has been completed")
            elif args.model_arch == 'noe':
                eval_noe()
                print("Evaluation of the NOE model has been completed")
            else:
                print("Model file does not exists")
        elif args.grid_search:
            train_narx_grid()
        elif args.pred_submission:
            if args.model_arch == 'narx':
                prediction_narx()
                print("Narx prediction submission done")
            elif args.model_arch == 'noe':
                prediction_noe()
                print("Noe prediction submission done")
        elif args.sim_submission:
            if args.model_arch == 'narx':
                simulation_narx()
                print("Narx simulation submission done")
            elif args.model_arch == 'noe':
                simulation_noe()
                print("Noe simulation submission done")
        elif args.grid_eval:
            eval_grid()



    # 3. POLICY LEARNING : Reinforcement Learning
    elif args.method == 'rl':
        # perform task
        if args.train:
            rlm = RLManager(env=args.env,
                            method=args.agent,
                            mode='train',
                            multi_target=args.multi_target)
            rlm.train(render=args.render)
        elif args.test:
            rlm = RLManager(env=args.env,
                            method=args.agent,
                            mode='test',
                            multi_target=args.multi_target,
                            model_path = args.load)
            rlm.simulate()

if __name__ == '__main__':
    __main__()
