"""
    @Author Jay Lee
    Some common utilities that help out with deep learning.
"""
try:
    import argparse
except ImportError:
    raise ImportError("Failed to import argparse module")


def parser_with_default_arguments(batch_size=128, lr=0.001):
    """
        Add basic arguments to the current python file
        :param batch_size:
        :param lr:
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to run experiment")
    parser.add_argument("-l", "--lr", type=float, default=lr, help="Learning rate")
    parser.add_argument("-b", "--batchsize", type=int, default=batch_size, help="The size of the batch when doing "
                                                                                "mini-batch training")
    return parser


if __name__ == "__main__":
    parser = parser_with_default_arguments()
    parser.parse_args()
    print(f"Epochs: {parser.epochs}")