"""CS69000-DPL - HW2 (Demo Program)

    run a demo for students to show what we expect see in submissions.

Author: I-Ta Lee
"""
import sys
import argparse
import logging
import torch
from my_neural_networks import activations


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='CS69000-DPL HW2 Demo')
    parser.add_argument('required_arg', metavar='REQUIRED_ARG',
                        help='An example argument to show you how to use argparse. Just put a random string.')

    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id to use. -1 means cpu (DEFAULT: -1)')

    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def bin_config(get_arg_func):
    # get arguments
    args = get_arg_func(sys.argv[1:])

    # set logger
    logger = logging.getLogger()

    # if you want to show debug message
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')

    # This shows you how to dump a log file for debugging
    fpath = 'log'
    fileHandler = logging.FileHandler(fpath)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return args


def main():
    """Demo How to customize a Python package

    This program shows you an illustration about how to write a main function along with a python package.
    We expect you to write your main program, e.g., hw1.py, in this manner.


    # Usages
    ---
    Type the following command to run this program:
            > python demo.py any_string

    argparse and logging are two substantial Python packages that we encourage you to study and use
    many good tutorials are online.


    # More Examples
    ---
    You should also try out the following commands to see the difference:
            > python demo.py -d any_string

    If you have GPUs available:
            > python demo.py -g 0 any_string
    """
    # An example of required arguments
    logging.info("your required argument is {}".format(args.required_arg))

    # As an example, a Sigmoid function is provided in the module "my_neural_networks/activations.py"
    # Here we simply create a random tensor and call it.
    X = torch.rand(4, 3) * 10

    # this shows you an option of using gpu
    # you need to specified the gpu id in the arguments
    if args.gpu_id != -1:
        X = X.cuda(args.gpu_id)

    logging.info("X = {}".format(X))
    # call a funciton in the module--activations.
    X_sigmoid = activations.sigmoid(X)

    logging.debug("This will not show up if you didn't specified -d as your command-line arguments.")
    logging.info("sigmoid(X) = {}".format(X_sigmoid))

    # More questions? post it on Piazza


if __name__ == '__main__':
    args = bin_config(get_arguments)
    main()
