# -*- coding: UTF-8 -*-
import argparse

import tensorflow as tf
import sys


"""
Created by xujinliang on 2018.9.15
"""



def main(argv=None):
    cifar10_pruning.maybe_download_and_extract()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/cifar10_train',
        help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
        '--pruning_hparams',
        type=str,
        default='',
        help="""Comma separated list of pruning-related hyperparameters""")
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000000,
        help='Number of batches to run.')
    parser.add_argument(
        '--log_device_placement',
        type=bool,
        default=False,
        help='Whether to log device placement.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
