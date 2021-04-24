#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:48:58 2021

@author: yujiang
"""
"""
for visualization of multi cam results
visualize tracking result on img

format of track file:
<frame id>, <object id>, <x>, <y>, <w>, <h>

"""

import argparse
import os
import cv2
import numpy as np
import skimage.io

from config import FPS, color_list, resize_ratio
from visualize import loadImgList, draw_caption

def config():
    a = argparse.ArgumentParser(description='multi cam vis')
    # path control
    a.add_argument('--result_file1', default='output/ctracker/train_cam034_synset/results/cam0_csv/intertraj/set05_mul.txt', type=str, help='path to tracking result file')    
    a.add_argument('--result_file2', default='output/ctracker/train_cam034_synset/results/cam4_csv/intertraj/set05_mul.txt', type=str, help='path to tracking result file')  
    # img mode
    a.add_argument('--img_dir', default='data/dataset5/cam4/set04/img1', type=str, help='path to image folder')

    # csv mode
    a.add_argument('--csvfile1', default='data/dataset5/ctracker/split19/cam0/set05.csv', type=str, help='path to image folder')
    a.add_argument('--csvfile2', default='data/dataset5/ctracker/split19/cam4/set05.csv', type=str, help='path to image folder')
    a.add_argument('--csvmode', action='store_true', help='whether to use csv file, instead of img_dir mode')
    a.add_argument('--root', default='data/dataset5', type=str, help='Path of root dir of dataset (then connect to img path in test csv)')
    
    args = a.parse_args()
    
    return args

def main(args):   
    # load img list from cam1
    img_list1, imgnames1, minframeid1 = loadImgList(args.csvfile1, args.root, args.img_dir, args.csvmode)        
    # load tracking result file
    track1 = np.genfromtxt(args.result_file1,delimiter=',',usecols=(1, 2,3,4,5)).astype(int)
    frameidxcol1 = np.genfromtxt(args.result_file1,delimiter=',',usecols=(0)).astype(int)

    # load img list from cam2
    img_list2, imgnames2, minframeid2 = loadImgList(args.csvfile2, args.root, args.img_dir, args.csvmode)             
    # load tracking result file
    track2 = np.genfromtxt(args.result_file2,delimiter=',',usecols=(1, 2,3,4,5)).astype(int)
    frameidxcol2 = np.genfromtxt(args.result_file2,delimiter=',',usecols=(0)).astype(int)    
    
    img_len = len(img_list1)
    H, W, _ = skimage.io.imread(img_list1[0]).shape    
    
    # resize to smaller img
    H, W = int(H/(resize_ratio*2)), int(W/resize_ratio)
    
    # prepare video
    cam1 = args.result_file1.split('/')[-3]
    cam2 = args.result_file2.split('/')[-3]
    output_file = os.path.join(args.result_file2[:-len(os.path.basename(args.result_file2))], cam1+cam2+'.mp4')
    videoWriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, (W, H))
    print('save video to '+output_file)
    
    for i in range(img_len):
        print('processing image: '+imgnames1[i]+' and '+imgnames2[i])
        img1= cv2.imread(img_list1[i])
        img2= cv2.imread(img_list2[i])
        
        frameidx1 = int(os.path.basename(img_list1[i]).split('.')[0][5:]) - minframeid1
        track_cur1 = track1[frameidxcol1==frameidx1]
        frameidx2 = int(os.path.basename(img_list2[i]).split('.')[0][5:]) - minframeid2
        track_cur2 = track2[frameidxcol2==frameidx2]
        
        for line in track_cur1:
            x1, y1, x2, y2 = line[1], line[2], line[1]+line[3], line[2]+line[4]
            trace_id = line[0]
            draw_trace_id = str(trace_id)+' '+ cam1
            draw_caption(img1, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
            cv2.rectangle(img1, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)
       
        for line in track_cur2:
            x1, y1, x2, y2 = line[1], line[2], line[1]+line[3], line[2]+line[4]
            trace_id = line[0]
            draw_trace_id = str(trace_id)+' '+ cam2
            draw_caption(img2, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
            cv2.rectangle(img2, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)
            
        img = cv2.resize(np.concatenate((img1, img2), axis=1), (W, H))

        videoWriter.write(img)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = config()
    
    main(args)
