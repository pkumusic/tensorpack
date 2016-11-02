#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', required=True)
    parser.add_argument('task', help='task to perform',
                        choices=['play','eval','train'], default='train')
    args=parser.parse_args()
    ENV_NAME = args.env
    print ENV_NAME