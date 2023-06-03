# System Imports
import __main__
import os
import argparse

# Local Relative Imports
from config.definitions import *
# from ann.model import train_ann, eval_ann
from gp.model import PendulumManager

###########  P A R A M E T E R S  ###########

# Define Arguments
parser = argparse.ArgumentParser()

# Usual arguments which are applicable for the whole script / top-level args
parser.add_argument('--verbose', help='Common top-level parameter',
                    action='store_true', required=False)

## General Arguments
parser.add_argument('--model-file', type = str, default = os.path.join(MODELS_DIR, "model.pth"),
                           required=False, help = 'Specify the model path to be stored/loaded')
parser.add_argument('--model-arch', type=str, choices=['narx, noe, ss'], default='narx',
                           required=False, help='Choose model architecture')

subparsers = parser.add_subparsers(title="actions")


# PARENT PARSER FOR SOME SUBPARSERS TO INHERIT THE ARGUMENTS IN ORDER TO BE SHARED, THUS WE DONT NEED
# TO SPECIFY EACH SHARED ARGUMENTS TO EACH SEPERATE SUBPARSER
parent_parser = argparse.ArgumentParser(description = 'put some description here...', add_help=False)
parent_parser.add_argument('--train', action='store_true', default=False, help='Train the model')
parent_parser.add_argument('--test', action='store_true', default=False, help='Train the model')


parser_method = subparsers.add_parser("method", parents=[parent_parser],
                                      description='The method parser', help='Method to be chosen (ANN or GP)')
parser_method.add_argument('--name', type = str, default="ann", help="name of the method to be used")

# parser_mode = subparsers.add_parser("mode", parents=[parent_parser], add_help=False,
#                                       description="The mode parser", help="Mode to be chosen (Train or Test)")
# parser_mode.add_argument("--action", action='store_true', help="action of the mode to be used")
# parser_mode.add_argument('--save', action='store_true', default=False, help='Save the model')

subparsers.required = True

def __main__():
    # Compile Arguments
    args = parser.parse_args()
    # print(parser.print_help())
    print(args)

    if args.train:
        if args.name == 'gp':
            PendulumManager()
        # elif args.name == 'ann':
            # train_ann()
            # print("Training of the model has been completed")
    elif args.test:
        if args.name == 'gp':
            PendulumManager()
        elif args.name == 'ann':
            # if eval_ann(args.mode_path, args.model_arch):
                # print("Evaluation of the model has been completed")
            # else:
                print("Model file does not exists")

if __name__ == '__main__':
    __main__()
