#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:48:58 2021

@author: yujiang
"""
"""

visualize tracking result on img

format of track file:
<frame id>, <object id>, <x>, <y>, <w>, <h>

"""

import argparse
import os
import cv2
import numpy as np
import skimage.io

from mov2static import computeP

# gobal param
FPS = 30
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), 
(0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]

'''
visualize tracking results and save to video

'''
def config():
    a = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')
    # path control
    a.add_argument('--result_file', default='output/ctracker/train_cam034_synset/results/cam4_csv/mulcam/set06.txt', type=str, help='path to tracking result file')    
    # img mode
    a.add_argument('--img_dir', default='data/dataset5/cam4/set04/img1', type=str, help='path to image folder')

    # csv mode
    a.add_argument('--csvfile', default='data/dataset5/ctracker/split19/cam4/set06.csv', type=str, help='path to image folder')
    a.add_argument('--csvmode', action='store_true', help='whether to use csv file, instead of img_dir mode')
    a.add_argument('--root', default='data/dataset5', type=str, help='Path of root dir of dataset (then connect to img path in test csv)')
    
    # default is <x, y, w, h>, switch to xy format: <x1, y1, x2, y2>
    a.add_argument('--xymode', action='store_true', help='whether to format <x1, y1, x2, y2>')
    
    # default is 2d, switch to 3d mode (project bbox to pitch)
    a.add_argument('--pitchmode', action='store_true', help='whether to use 3d mode')
    a.add_argument('--pitch', default='data/soccer_pitch/soccer_pitch_grass.jpg', type=str, help='path to soccer pitch image')
    a.add_argument('--pitchhomoroot', default='data/soccer_pitch/homography', type=str, help='path to soccer pitch image')
    a.add_argument('--calib_file', default='data/calibration_results/0125-0135/CAM1/calib.txt', type=str, help='path to calibration file')
    a.add_argument('--n', default=0, help='n-th frame as reference', type=int)
    
    args = a.parse_args()
    
    return args

def draw_caption(image, box, caption, color):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
def draw_3d(proj, x, y):
    cp = np.dot(proj, np.array([x, y, 1]))
    cp = cp / cp[-1]
    return cp

def main(args):   
    # load img list
    minframeid = 0
    if args.csvmode:
        imgpaths = np.genfromtxt(args.csvfile, delimiter=',',usecols=(0), dtype=str)
        img_list = [os.path.join(args.root, imgpath) for imgpath in imgpaths] 
        # find unique img path
        img_list = list(set(img_list))
        # sort
        frameid = [int(os.path.basename(x).split('.')[0][5:]) for x in img_list]
        minframeid = min(frameid) - 1
        img_list = [x for _, x in sorted(zip(frameid, img_list))]
        imgnames = [os.path.basename(x) for x in img_list]
        
    else:
        imgnames = os.listdir(args.img_dir)
        img_list = [os.path.join(args.img_dir, imgname) for imgname in imgnames]
        # sort
        frameid = [int(os.path.basename(x).split('.')[0][5:]) for x in img_list]
        minframeid = min(frameid) - 1
        img_list = [x for _, x in sorted(zip(frameid, img_list))]
        imgnames = [os.path.basename(x) for x in img_list]
        
    img_len = len(img_list)
    H, W, _ = skimage.io.imread(img_list[0]).shape
    
    # load tracking result file
    track = np.genfromtxt(args.result_file,delimiter=',',usecols=(1, 2,3,4,5)).astype(int)
#    track = np.genfromtxt(args.result_file,delimiter=',',usecols=(1, 2,3,4,5, 7)).astype(int)
    frameidxcol = np.genfromtxt(args.result_file,delimiter=',',usecols=(0)).astype(int)
    
    # prepare video
    if args.pitchmode:
        output_file = os.path.join(args.result_file[:-len(os.path.basename(args.result_file))], os.path.basename(args.result_file.split('.')[0]+'_3d.mp4'))
    else:
        output_file = os.path.join(args.result_file[:-len(os.path.basename(args.result_file))], os.path.basename(args.result_file.split('.')[0]+'.mp4'))
    videoWriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, (W, H))
    print('save video to '+output_file)
    
    #######
    #  3D #
    #######
    setname = args.calib_file.split('/')[-2]
    if args.pitchmode:
        # load calib and image file list
        calib = np.genfromtxt(args.calib_file,delimiter=',',usecols=(1,2,3,4,5,6))
        imgname = np.genfromtxt(args.calib_file,delimiter=',',usecols=(7), dtype=str)
        framecalib = [int(x.split('.')[0][5:]) for x in imgname]
        
        # load homography matrix
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
        if setname == 'CAM1':
            args.n = -1
        P1 = computeP(calib[args.n], cx, cy)
        
        # read homo for soccer pitch
        homo_pitch = np.load(os.path.join(args.pitchhomoroot, setname+'.npy'))
        
        for i in range(img_len):
            print('processing image (3d mode): '+imgnames[i])
            # create a background image
            img = cv2.imread(args.pitch)
#            img = cv2.warpPerspective(img, T, (WIDTH, HEIGHT))
            
#            img = skimage.io.imread(img_list[i])
            frameidx = int(os.path.basename(img_list[i]).split('.')[0][5:]) - minframeid
            track_cur = track[frameidxcol==frameidx]
            
            # find index in calib file
            if not frameidx in framecalib:
                continue
            else:
                # project to soccer pitch
                calibline = calib[framecalib.index(frameidx)]
            
            # project to soccer pitch
            P = computeP(calibline, cx, cy)      
            # compute homography
            H = np.dot(P1, np.linalg.pinv(P))       
            proj = np.dot(homo_pitch, H)

            # project to 1st frame (frame0.jpg) x1 = Hx
#            img = cv2.warpPerspective(img, proj, (WIDTH, HEIGHT))
            
            for line in track_cur:
                if args.xymode:
                    x1, y1, x2, y2 = line[1], line[2], line[3], line[4]
                else:
                    x1, y1, x2, y2 = line[1], line[2], line[1]+line[3], line[2]+line[4]
                trace_id = line[0]
                draw_trace_id = str(trace_id)
                
                # compute center point (x as horizontal axis)
                cxp, cyp = int((x1+x2)/2), y2
                cp = draw_3d(proj, cxp, cyp)
                cv2.circle(img, (int(cp[0]), int(cp[1])), radius=10, color=color_list[trace_id % len(color_list)], thickness=-1)
#                cp1 = draw_3d(proj, x1, y1)
#                cp2 = draw_3d(proj, x1, y2)
#                cp3 = draw_3d(proj, x2, y1)
#                cp4 = draw_3d(proj, x2, y2)
#                cv2.circle(img, (int(cp1[0]), int(cp1[1])), radius=2, color=color_list[trace_id % len(color_list)], thickness=-1)
#                cv2.circle(img, (int(cp2[0]), int(cp2[1])), radius=2, color=color_list[trace_id % len(color_list)], thickness=-1)
#                cv2.circle(img, (int(cp3[0]), int(cp3[1])), radius=2, color=color_list[trace_id % len(color_list)], thickness=-1)
#                cv2.circle(img, (int(cp4[0]), int(cp4[1])), radius=2, color=color_list[trace_id % len(color_list)], thickness=-1)
            print('save to '+os.path.join(args.result_file[:-len(os.path.basename(args.result_file))], 'img_'+setname, 'img_'+str(i)+'.jpg'))
            cv2.imwrite(os.path.join(args.result_file[:-len(os.path.basename(args.result_file))], 'img_'+setname, 'img_'+str(i)+'.jpg'), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
#            videoWriter.write(img)
        cv2.waitKey(0)
    
    
    #######
    #  2D #
    #######
    else:      
        for i in range(img_len):
            print('processing image (2d mode): '+imgnames[i])
            img = cv2.imread(img_list[i])
            frameidx = int(os.path.basename(img_list[i]).split('.')[0][5:]) - minframeid
            track_cur = track[frameidxcol==frameidx]
            
            for line in track_cur:
                if args.xymode:
                    x1, y1, x2, y2 = line[1], line[2], line[3], line[4]
                else:
                    x1, y1, x2, y2 = line[1], line[2], line[1]+line[3], line[2]+line[4]
                trace_id = line[0]
                draw_trace_id = str(trace_id)
                draw_caption(img, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)
            videoWriter.write(img)
        cv2.waitKey(0)

if __name__ == '__main__':
    args = config()
    
    main(args)
