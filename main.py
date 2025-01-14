from train import *
from test import *
from model_opt import *


def main():
    if model_opt['state'] == 'train':
        train(model_opt['epoch'], model_opt['attention'])
    elif model_opt['state'] == 'test':
        test()
    else:
        outputImage("2011_001005.jpg")


if __name__ == '__main__':
    main()
