#!/opt/conda/bin/python3

from __future__ import print_function

import argparse

import torch


def main():
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)', required=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)', required=True)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    print(torch.cuda.get_device_name(0))
        
if __name__ == '__main__':
    main()
