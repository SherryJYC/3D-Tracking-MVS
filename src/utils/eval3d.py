#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:43:40 2021

@author: yujiang
"""

"""
evaluation of 3D positions

from tracking result <frame id, obj id, x, z>

execution file

"""
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

def config():
    a = argparse.ArgumentParser()

    a.add_argument("--pred", help="track result of cam", default='output/pitch/cam1_right_team.txt')
    a.add_argument("--gt", help="ground truth", default='data/0125-0135/3d_gt/gt_pitch.txt')

    a.add_argument('--fixcam', help="use fix cam mode", action='store_true')

    args = a.parse_args()
    return args


def main(args):
    # load pred and gt (t, x, z)
    pred = np.genfromtxt(args.pred, delimiter=',', usecols=(0, 2, 3))
    gt = np.genfromtxt(args.gt, delimiter=',', usecols=(0, 2, 3))

    if args.fixcam:
        pred[:, 1]*= -1

    frames = np.unique(pred[:, 0])
    errors = np.empty((1, ))

    for frame in frames:
        cur_pred = pred[pred[:, 0] == frame]
        cur_gt = gt[gt[:, 0] == frame]

        # compute cost
        cost = np.full((len(cur_pred), len(cur_gt)), np.inf)
        for i, p in enumerate(cur_pred):
            for j, g in enumerate(cur_gt):
                cost[i, j] = np.linalg.norm(p[1:] - g[1:])
        
        # hungarian algorithm to find match
        row_ind, col_ind = linear_sum_assignment(cost)
        errors = np.concatenate((errors, cost[row_ind, col_ind].reshape((-1, ))))


    print('reprojection error: mean {}, std {}'.format(np.mean(errors), np.std(errors)))

if __name__ == '__main__':
    main(config())