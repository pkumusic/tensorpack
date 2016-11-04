#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Authors: Music, Tian, Jing
# This file is used to sample images for pre-training object recognition

from tensorpack.RL.expreplay import ExpReplay
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.common import MapPlayerState, PreventStuckPlayer, LimitLengthPlayer
from tensorpack.RL.history import HistoryFramePlayer
import cv2
import numpy as np

ENV_NAME = 'Freeway-v0'
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4

def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1])
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    if not train:
        pass
        #pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        #pl = PreventStuckPlayer(pl, 30, 1) #TODO: I think we don't need this in freeway. Is any bug in this code? didn't see repeated actions.
    pl = LimitLengthPlayer(pl, 40000)
    return pl


if __name__ == '__main__':
    #from .atari import AtariPlayer
    #predictor = lambda x: np.array([1,1,1,1])
    #player = AtariPlayer(sys.argv[1], viz=0, frame_skip=10, height_range=(36, 204))
    player = get_player()
    player.action(2)
    #player.restart_episode()
    # Original image: (210 * 160 * 3) by print player.current_state().shape
    print player.current_state().shape
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(player.current_state())
    plt.show()



    #
    #
    # E = ExpReplay(predictor,
    #         player=player,
    #         num_actions=player.get_action_space().num_actions(),
    #         populate_size=1001,
    #         history_len=4)
    # E._init_memory()
    #
    # for k in E.get_data():
    #     import IPython as IP;
    #     IP.embed(config=IP.terminal.ipapp.load_default_config())
    #     pass
    #     #import IPython;
    #     #IPython.embed(config=IPython.terminal.ipapp.load_default_config())
    #     #break