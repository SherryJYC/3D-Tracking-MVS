#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:43:40 2021

@author: yujiang
"""

"""
incrementally

combine cameras from tracking result <frame id, obj id, x, z>

execution file

"""
import argparse
import os

from mcTracker import Camera, Pitch
from mcTracker_reid import Camera_reid, Pitch_reid

def config():
    a = argparse.ArgumentParser()
    a.add_argument('--doreid', action='store_true', help='whether to use reid')
    # a.add_argument("--camlist", nargs="+", default=['data/tracks/cam1_filtered_team_pitch.txt', 
    #                                         'data/tracks/right_filtered_team_pitch.txt'])
    a.add_argument("--camlist", nargs="+", default=['data/tracks/fixcam/EPTS_1_pitch.txt', 
                                            'data/tracks/fixcam/EPTS_2_pitch.txt', 
                                            'data/tracks/fixcam/EPTS_3_pitch.txt', 
                                            'data/tracks/fixcam/EPTS_4_pitch.txt', 
                                            'data/tracks/fixcam/EPTS_5_pitch.txt',
                                            'data/tracks/fixcam/EPTS_6_pitch.txt',
                                            'data/tracks/fixcam/EPTS_7_pitch.txt',
                                            'data/tracks/fixcam/EPTS_8_pitch.txt'])

    args = a.parse_args()
    return args

def pairwiseMatch(input_cam1, input_cam2, doreid=False):

    output_file = os.path.basename(input_cam1) + '_' + os.path.basename(input_cam2) +'.txt'
    output_dir = os.path.join(input_cam1[:-len(os.path.basename(input_cam1))], 'results')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = os.path.join(output_dir, output_file)

    if doreid:
        print('reid matching ...')
        # load camera
        cam1 = Camera_reid(input_cam1, os.path.basename(input_cam1))
        cam2 = Camera_reid(input_cam2, os.path.basename(input_cam2))

        pitch = Pitch_reid(output=output_file)
    
    else:
        print('no reid matching ...')
        # load camera
        cam1 = Camera(input_cam1, os.path.basename(input_cam1))
        cam2 = Camera(input_cam2, os.path.basename(input_cam2))

        pitch = Pitch(output=output_file)

    pitch.add_cam(cam1)
    pitch.add_cam(cam2)
    print('find {} cameras in frame {} to {}'.format(len(pitch.cam_list), pitch.tstart, pitch.tend))

    # main process
    pitch()

    return output_file


def main(args):
    # tree builder
    input_cams = args.camlist

    level = 0
    while (True):
        print('--- Level {}: combine {} cams ---'.format(level, len(input_cams)))
        newcams = []
        for camid in range(len(input_cams)):
            if camid%2 == 1:
                continue 
            if camid+1 >= len(input_cams):
                break

            # combine two cam
            newcam = pairwiseMatch(input_cams[camid], input_cams[camid+1], args.doreid)
            newcams.append(newcam)

        # termination condition
        if len(newcams) == 1:
            print('=== final output file {} ==='.format(newcam))
            break

        # continue iteration
        input_cams = newcams
        level += 1

if __name__ == '__main__':
    main(config())