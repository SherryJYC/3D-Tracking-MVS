#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import skvideo.io
from skvideo.io import FFmpegWriter
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from iou_tracker import track_iou
from util import load_mot
from natsort import natsorted
import os
import glob

def render_frame(im, tracks):
    for id_, track in tracks.items():
        xmin, ymin, xmax, ymax = track
        xmin = int(xmin)
        ymin = int(ymin)
        width = int(xmax - xmin)
        height = int(ymax - ymin)
        label = str(id_)
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im
        


def parse_tracks(trakcs):
    tracks_per_frame = {}
    for id_, track in enumerate(tracks):
        bboxes = track['bboxes']
        score = track['max_score']
        start_frame = track['start_frame']
        for i, bbox in enumerate(bboxes):
            frame = start_frame + i
            if frame not in tracks_per_frame:
                tracks_per_frame[frame] = {}
            tracks_per_frame[frame][id_] = bbox
    return tracks_per_frame


     
def write_video(INPUT_VIDEO, tracks_per_frame, OUTPUT_VIDEO):
    #Reading from images under the given directory
    output_video = FFmpegWriter(OUTPUT_VIDEO)


    if os.path.isdir(INPUT_VIDEO):
        img_paths = natsorted(glob.glob(INPUT_VIDEO+"/*.jpg"))

        for i, img_path in enumerate(img_paths, start=1):
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracks = tracks_per_frame.get(i, {})
            output_frame = render_frame(frame, tracks)
            output_video.writeFrame(output_frame)
            print("Writen Frame: {}".format(i))


    #Reading from a video   
    else:   
        print("Reading Video {}".format(INPUT_VIDEO))
        input_video = skvideo.io.vread(INPUT_VIDEO)
        print("Reading Finished")           
        for i, frame in enumerate(input_video, start=1):      
            tracks = tracks_per_frame.get(i, {})
            output_frame = render_frame(frame, tracks)
            output_video.writeFrame(output_frame)
            print("Writen Frame: {}".format(i))



    output_video.close()
    
    
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8



if __name__ == '__main__':
   
    #INPUT_VIDEO = "./MOT17-04-SDP.mp4"
    INPUT_PATH = "MOT17/train/MOT17-04-SDP/img1"
    INPUT_DETECTION =  "MOT17/train/MOT17-04-SDP/det/det.txt"
    OUTPUT_VIDEO= "./result.mp4"

    detections = load_mot(INPUT_DETECTION)

    sigma_l = 0
    sigma_h = 0.5
    sigma_iou = 0.5
    t_min = 2
    tracks = track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
    tracks_per_frame = parse_tracks(tracks)
    write_video(INPUT_PATH, tracks_per_frame, OUTPUT_VIDEO)