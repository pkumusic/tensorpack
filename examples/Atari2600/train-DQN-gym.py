#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
import argparse
import os
from tensorpack.predict.common import PredictConfig
from tensorpack import *
from tensorpack.models.model_desc import ModelDesc
from tensorpack.train.config import TrainConfig
from tensorpack.tfutils.common import *
from tensorpack.callbacks.base import Callback

STEP_PER_EPOCH = 6000

class Model(ModelDesc):
    pass
def get_config():
    logger.auto_set_dir()

    M = Model()
    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    return TrainConfig(
        #dataset = ?, # A dataflow object for training
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callback(),

        session_config = get_default_sess_config(0.6),  # Tensorflow default session config consume too much resources.
        model = M,
        step_per_epoch=STEP_PER_EPOCH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('-l','--load', help='load model')
    parser.add_argument('-e','--env', help='env', required=True)
    parser.add_argument('-t','--task', help='task to perform',
                        choices=['play','eval','train'], default='train')
    args=parser.parse_args()
    ENV_NAME = args.env

    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task == 'train':
        config = get_config()

    else:
        pass
