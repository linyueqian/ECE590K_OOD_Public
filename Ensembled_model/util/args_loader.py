import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='CIFAR-100')
    parser.add_argument('--out-datasets', default=['SVHN', 'MNIST', 'Texture'], type=list)#
    parser.add_argument('--name', default="lenet", type=str, help='neural network name and training set')
    parser.add_argument('--model-arch', default='lenet', type=str, help='model architecture [resnet18]')
    parser.add_argument('--optim', default='Bayessian', type=str, help='Adam')
    parser.add_argument('--threshold', default=1.0, type=float, help='sparsity level')
    parser.add_argument('--method', default='msp', type=str, help='odin mahalanobis CE_with_Logst')
    parser.add_argument('--knn_param', default=50, type=int, help='odin mahalanobis CE_with_Logst')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')

    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.set_defaults(argument=True)
    args = parser.parse_args()
    print(args)
    return args
get_args()