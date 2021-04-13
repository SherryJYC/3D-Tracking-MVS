#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:01:35 2021

@author: yujiang
"""

'''

convert frames of moving cameras to static setting (refer to 1st frame cam pose)

'''

import argparse
import math
import numpy as np
import os
import skimage.io
import cv2

def config():
    a = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')
    a.add_argument('--calib_file', default='data/calibration_results/0125-0135/CAM1/calib.txt', type=str, help='path to calibration file')
    a.add_argument('--img_dir', default='data/0125-0135/CAM1/img', help='Path to original image folder')
    a.add_argument('--output_dir', default='data/0125-0135/CAM1/img_static', help='Output path to projected image folder')
    
    a.add_argument('--n', default=0, help='n-th frame as reference', type=int)
    
    args = a.parse_args()
    
    return args

def Rx(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    A = np.array([[1, 0, 0],
                  [0, rcos, -rsin],
                  [0, rsin, rcos]])
    return A


def Ry(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    #A = np.array([[rcos, 0, rsin],
    #              [0, 1, 0],
    #              [-rsin, 0, rcos]])
    K = np.array([[rcos, 0, -rsin],
                  [0, 1, 0],
                  [rsin, 0, rcos]])
    return K

def computeP(line, cx, cy):
    P = np.empty([3,4])
    
    theta, phi, f, Cx, Cy, Cz = line
    R = Rx(phi).dot(Ry(theta).dot(np.array([[1,0,0],[0,-1,0],[0,0,-1]])))
    T = -R.dot(np.array([[Cx], [Cy], [Cz]]))
    K = np.eye(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, cx, cy
    P = np.dot(K, np.hstack((R, T))) 

    return P

def main(args):
    # load calib and image file list
    calib = np.genfromtxt(args.calib_file,delimiter=',',usecols=(1,2,3,4,5,6))
    imgname = np.genfromtxt(args.calib_file,delimiter=',',usecols=(7), dtype=str)
 
    img_list = [os.path.join(args.img_dir, one_imgname) for one_imgname in imgname]
    frameidx = list(map(int, [x.split('/')[-1].split('.')[0][5:] for x in img_list])) 
    img_list = [x for _,x in sorted(zip(frameidx,img_list))]
    
    assert len(img_list) == len(calib), "total length of image frames and calibration file unmatched !"
    
    # homography matrix
    homo = np.load(os.path.join(args.img_dir[:-len(args.img_dir.split('/')[-1])], 'homo.npy'))
    
    # projection param from proj_config.txt
    config_file = os.path.join(args.img_dir.split('/')[0], args.img_dir.split('/')[1], 'proj_config.txt')
    param = np.genfromtxt(config_file, delimiter=',').astype(int)
    
    HEIGHT, WIDTH = param[0], param[1]
    cx,cy = WIDTH/2., HEIGHT/2.
    woffset = param[2]
    hoffset = param[3]
    scale= param[4]
    
    WIDTH+=int(woffset*3)
    HEIGHT+=int(woffset*2)
    
    # put reference frame at center
    T = np.eye(3,3)*scale
    T[-1,-1] = 1
    T[0,-1], T[1,-1] = woffset, hoffset
    
    # project each image to nth frame 
    P1 = computeP(calib[args.n], cx, cy)
    
    for i, line in enumerate(calib):
        print('projecting image frame {}'.format(i))
        img = skimage.io.imread(img_list[i])  
        P = computeP(line, cx, cy)      
              
        # compute homography
        H = np.dot(P1, np.linalg.pinv(P))       
        
        # project to 1st frame (frame0.jpg) x1 = Hx
        img = cv2.warpPerspective(img, np.dot(T, np.dot(homo, H)), (WIDTH, HEIGHT))
        
        # write to file
        cv2.imwrite(os.path.join(args.output_dir, imgname[i]), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    return

if __name__ == '__main__':
    args = config()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    main(args)