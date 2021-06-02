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
from matplotlib import pyplot as plt
import seaborn as sns

def config():
    a = argparse.ArgumentParser()

    a.add_argument("--pred", help="track result of cam", default='data/tracks/fixcam/results/results/EPTS_1_pitch.txt_EPTS_2_pitch.txt.txt_EPTS_3_pitch.txt_EPTS_4_pitch.txt.txt.txt')
    a.add_argument("--gt", help="ground truth", default='data/fixedcam/gt_pitch_550.txt')

    a.add_argument("--predlist", nargs="+", default=[
        'data/tracks/fixcam/results/EPTS_1_pitch.txt_EPTS_2_pitch.txt.txt',
    'data/tracks/fixcam/results/EPTS_3_pitch.txt_EPTS_4_pitch.txt.txt', 
    'data/tracks/fixcam/results/EPTS_5_pitch.txt_EPTS_6_pitch.txt.txt', 
    'data/tracks/fixcam/results/EPTS_7_pitch.txt_EPTS_8_pitch.txt.txt',
    'data/tracks/fixcam/results/results/EPTS_1_pitch.txt_EPTS_2_pitch.txt.txt_EPTS_3_pitch.txt_EPTS_4_pitch.txt.txt.txt', 
    'data/tracks/fixcam/results/results/EPTS_5_pitch.txt_EPTS_6_pitch.txt.txt_EPTS_7_pitch.txt_EPTS_8_pitch.txt.txt.txt',
    'data/tracks/fixcam/results/results/EPTS_1_pitch.txt_EPTS_2_pitch.txt.txt_EPTS_3_pitch.txt_EPTS_4_pitch.txt.txt.txt'])

    # if x = -x
    a.add_argument('--fixcam', help="use fix cam mode", action='store_true')
    a.add_argument('--boxplot', help="use box plot mode", action='store_true')

    args = a.parse_args()
    return args

def getStat(pred_file, gt_file, fixcam=True):
    # load pred and gt (t, x, z)
    pred = np.genfromtxt(pred_file, delimiter=',', usecols=(0, 2, 3))
    gt = np.genfromtxt(gt_file, delimiter=',', usecols=(0, 2, 3))

    if fixcam:
        pred[:, 1]*= -1

    frames = np.unique(pred[:, 0])
    errors = None

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

        if errors is None:
            errors = cost[row_ind, col_ind].reshape((-1, ))
        else:
            errors = np.concatenate((errors, cost[row_ind, col_ind].reshape((-1, ))))

    print('reprojection error: mean {}, std {}, min {}'.format(np.mean(errors), np.std(errors), np.min(errors)))

    return errors

def main(args):
    if not args.boxplot:
        errors = getStat(args.pred, args.gt, args.fixcam)
    else:
        all_errors = []
        for p in args.predlist:
            errors = getStat(p, args.gt, args.fixcam)
            all_errors.append(errors)

        # draw box plot
        fig, ax = plt.subplots()
        sns.boxplot(data=all_errors, showfliers=False, palette='Set3')
        ax.set_title('Boxplot of 3D position errors of tracking results (8 fixed cameras)')
        ax.set_ylabel('Error (m)')
        ax.set_xticklabels(['12', '34', '56', '78', '1-4', '5-8', '1-8'])
        plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
        plt.setp(ax.lines, color='k')
        plt.show()
       

if __name__ == '__main__':
    main(config())