#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
import argparse
import os
from tensorpack.predict.common import PredictConfig
from tensorpack import *
from tensorpack.models.model_desc import ModelDesc, InputVar
from tensorpack.train.config import TrainConfig
from tensorpack.tfutils.common import *
from tensorpack.callbacks.group import Callbacks
from tensorpack.callbacks.stat import StatPrinter
from tensorpack.callbacks.common import ModelSaver
from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter

STEP_PER_EPOCH = 6000

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
IMAGE_SHAPE3 = IMAGE_SIZE + (FRAME_HISTORY, ) # one state input


NUM_ACTIONS = 4 #TODO: Generate it automatically later.


class Model(ModelDesc):
    def _get_input_vars(self):

        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'next_state'),
                InputVar(tf.bool, (None,), 'isOver')]
        pass

    def _get_DQN_prediction(self, image):
        pass

    def _build_graph(self, inputs):
        state, action, reward, next_state, isOver = inputs
        predict_value = self._get_DQN_prediction() # N * NUM_ACTIONS #TODO: If we need self. here
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0) # N * NUM_ACTION
        pred_action_value = tf.reduce_sum(predict_value * action_onehot, 1) # N,



        pass





def get_config():
    logger.auto_set_dir()
    M = Model()
    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    return TrainConfig(
        #dataset = ?, # A dataflow object for training
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([StatPrinter(), ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',[(80, 0.0003), (120, 0.0001)]) # No interpolation
            # TODO: Some other parameters

            ]),

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
