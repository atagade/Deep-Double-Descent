import argparse
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
    parser.add_argument("--num_hidden_units", type=int, default=10, help="number of hidden units")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus *per machine*")

    return parser