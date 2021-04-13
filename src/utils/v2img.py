#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:20:03 2021

@author: yujiang
"""

import sys
import argparse
import os
import cv2

def extractImages(pathIn, pathOut, splitnum):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    print('fps: '+str(vidcap.get(cv2.CAP_PROP_FPS)))
    success,image = vidcap.read()
    success = True
    cv2.imwrite( pathOut + "/image"+ str(int(1+count/splitnum)).zfill(4)+".png", image) 

    while success:
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        if count%splitnum == 0:
            cv2.imwrite( pathOut + "/image"+ str(int(2+count/splitnum)).zfill(4)+".png", image)  
            print("/image"+ str(int(2+count/splitnum)).zfill(4)+".png")
        count = count + 1
    print(count)

if __name__=="__main__":
    
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video", default='./data/dataset3/cam2/cam2.mp4')
    a.add_argument("--pathOut", help="path to images", default='./data/dataset3/cam2/img')
    a.add_argument("--splitnum", help="save a image every n frame", default=10, type=int)
    
    args = a.parse_args()
    print(args)
    if not os.path.exists(args.pathOut):
        os.mkdir(args.pathOut)
    extractImages(args.pathIn, args.pathOut, args.splitnum)