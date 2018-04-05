# config of the model
# use python argparser
# python 3.6.2
import argparse

def get_args():
    # init parser name
    parser = argparse.ArgumentParser(description='MNIST')
    # for cmd line autotest
    # name
    # type
    # default :value
    # help
    # batch-size
    parser.add_argument('--batch-size', type=int, default=100, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=100, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs in training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum for SGD optimizer')
    parser.add_argument('--seed', type=float, default=6, help='random seed, 6 as lucky number')
    parser.add_argument('--log_interval', type=int, default=300, help='log interval batch index in training')
    parser.add_argument('--test_interval', type=int, default=5, help='log interval in testing')

    args = parser.parse_args()

    return args

class ModelConfiger:
    def __init__(self):
        self.n_hiddens = [512, 256, 128, 64]
        #self.n_hiddens = [256, 128, 64]
        self.type_act = 'relu'
        self.rate_dropout = 0.2

class ModelConfigerAutoencoder:
    def __init__(self):
        self.n_hiddens = [512, 256, 128, 64]
        self.type_act = 'relu'
        self.rate_dropout = 0.2
        self.n_middle = 10