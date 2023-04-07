import argparse
import os
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run MNIST_FCNN from Belkin et al.: python train.py --dataset MNIST --model FCNN --num_hidden_units 10 --num_gpus 1 
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="MNIST", help="dataset to train on")
    parser.add_argument("--model", type=str, default="FCNN", help="model to use")
    parser.add_argument("--width", type=int, default=10, help="width of ResNet")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus *per machine*")

    return parser

def add_to_json(path, nparameters, train_loss, test_loss):

    if not os.path.isfile(path):
        log = []
        with open(path, 'w+') as f:
            pass
    else:
        with open(path, 'r') as f:
            log = json.load(f)
        
    log.append({"nparameters": nparameters, "training_loss": train_loss, "test_loss": test_loss})     
    
    with open(path, 'w') as f:
        json.dump(log, f, indent = 4, separators = (',',': '))