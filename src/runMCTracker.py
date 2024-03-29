#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 2 16:43:40 2021
@author: yujiang
"""

"""
combine two cameras from tracking result <frame id, obj id, x, z>
execution file
"""
import argparse
import os

from mcTracker import Camera, Pitch
from mcTracker_reid import Camera_reid, Pitch_reid

def config():
    a = argparse.ArgumentParser()
    # a.add_argument("--input_cam1", help="track result of cam1", default='data/tracks/16m_cam1_pitch.txt')
    # a.add_argument("--input_cam2", help="track result of cam2", default='data/tracks/16m_right_pitch.txt')
    a.add_argument("--input_cam1", help="track result of cam1", default='data/tracks/movingcam/cam1_filtered_team_pitch_reid.txt')
    a.add_argument("--input_cam2", help="track result of cam2", default='data/tracks/movingcam/right_filtered_team_pitch_reid.txt')
    a.add_argument("--output_file", help="output file name", default='cam1_right_team.txt')

    a.add_argument('--doreid', action='store_true', help='whether to use reid')

    args = a.parse_args()
    return args


def main(args):
    if args.doreid:
        # load camera
        cam1 = Camera_reid(args.input_cam1, 'cam1')
        cam2 = Camera_reid(args.input_cam2, 'right')

        # create pitch
        output_file = os.path.join(args.input_cam1[:-len(os.path.basename(args.input_cam1))], args.output_file)
        pitch = Pitch_reid(output=output_file)
    else:
        # load camera
        cam1 = Camera(args.input_cam1, 'cam1')
        cam2 = Camera(args.input_cam2, 'right')

        # create pitch
        output_file = os.path.join(args.input_cam1[:-len(os.path.basename(args.input_cam1))], args.output_file)
        pitch = Pitch(output=output_file)

    pitch.add_cam(cam1)
    print(pitch.tend)
    pitch.add_cam(cam2)
    print(pitch.tend)
    print('find {} cameras in frame {} to {}'.format(len(pitch.cam_list), pitch.tstart, pitch.tend))

    # main process
    pitch()


if __name__ == '__main__':
    main(config())
