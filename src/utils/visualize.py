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

from mov2static import computeP, Rx, Ry

# gobal param
FPS = 30
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), 
(0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]

'''
visualize tracking results and save to video

'''
def config():
    a = argparse.ArgumentParser(description='Simple script for visualize')
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
    a.add_argument('--vis_calib_file', default=None, type=str, help='path to the calibration file of the camera to be visualized')
    a.add_argument('--vis_result_file', default=None, type=str, help='path to the tracking result file of the camera to be visualized')
    
    args = a.parse_args()
    
    return args

def draw_caption(image, box, caption, color):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
def draw_3d(proj, x, y):
    cp = np.dot(proj, np.array([x, y, 1]).reshape(3,1))
    cp = cp / cp[-1]
    return cp

def compose_matrix(line, cx, cy):
    theta, phi, f, Cx, Cy, Cz = line
    R = Rx(phi).dot(Ry(theta).dot(np.array([[1,0,0],[0,-1,0],[0,0,-1]])))
    T = -R.dot(np.array([[Cx], [Cy], [Cz]]))
    K = np.eye(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, cx, cy

    return K, R, T

def homogenize(x):
    return np.vstack((x,1))

def compute_homo(calib1, calib2, cx, cy):

    # print(Trans)
    # print(K2)

    K1, R1, T1 = compose_matrix(calib1, cx, cy)
    K2, R2, T2 = compose_matrix(calib2, cx, cy)

    # mc1toc2 = np.dot(np.vstack((np.hstack((R2,T2)),np.array([0,0,0,1]))),np.vstack((np.hstack((R1.T,np.dot(-R1.T,T1))),np.array([0,0,0,1]))))
    # mc1toc2 = mc1toc2[:3,:]
    # H = K2.dot(mc1toc2.dot(np.linalg.inv(K1)))

    H = K2.dot(np.hstack((R2,T2))).dot(np.linalg.pinv(np.hstack((R1,T1)))).dot(np.linalg.inv(K1))

    return H

def homo_from_pose(calib1, calib2, cx, cy):

    K1, R1, t1 = compose_matrix(calib1, cx, cy)
    K2, R2, t2 = compose_matrix(calib2, cx, cy)

    R_1to2 = R2.dot(R1.T)
    t_1to2 = R2.dot(-(R1.T).dot(t1)) + t2

    normal = np.zeros((3,1))
    normal[1] = 1
    normal1 = R1.dot(normal)

    origin = np.zeros((3,1))
    origin1 = R1.dot(origin) + t1
    d_inv1 = 1/np.inner(normal1.ravel(),origin1.ravel())
    print(t_1to2)
    print(1/d_inv1)

    homo_eu = R_1to2 + d_inv1*t_1to2.dot(normal1.T)

    homo = K2.dot(homo_eu.dot(np.linalg.inv(K1)))

    homo /= homo[2,2]

    return homo

def skew(e):
    return np.array([[0,e[2],-e[1]],[-e[2],0,e[0]],[e[1],-e[0],0]])

def enforce_F(F):
    U,S,V = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2]=0
    return U.dot(S.dot(V))

def fundamental_from_pose(calib1, calib2, cx, cy):
    # theta1, phi1, f1, Cx1, Cy1, Cz1 = calib1
    # R1 = Rx(phi1).dot(Ry(theta1).dot(np.array([[1,0,0],[0,-1,0],[0,0,-1]])))
    # T1 = np.array([[Cx1], [Cy1], [Cz1]])
    # K1 = np.eye(3, 3)
    # K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2] = f1, f1, cx, cy

    # theta2, phi2, f2, Cx2, Cy2, Cz2 = calib2
    # R2 = Rx(phi2).dot(Ry(theta2).dot(np.array([[1,0,0],[0,-1,0],[0,0,-1]])))
    # T2 = np.array([[Cx2], [Cy2], [Cz2]])
    # K2 = np.eye(3, 3)
    # K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2] = f2, f2, cx, cy
    # P2 = np.dot(K, np.hstack((R2, -R2.dot(T2))))
    P1 = computeP(calib1, cx, cy)
    P2 = computeP(calib2, cx, cy)
    T2 = np.array(calib2[3:]).reshape((3,1))

    e = P1.dot(homogenize(T2)).ravel()
    # e /= e[2]

    F = skew(e).dot(P1.dot(np.linalg.pinv(P2)))

    return enforce_F(F)

def draw_line(img, line, text, color):
    _, c, _ = img.shape
    
    x0, y0 = map(int, [0, -line[2]/line[1]])
    x1, y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]])
    cv2.line(img, (x0, y0), (x1, y1), color, 1)
    # add text
    cv2.putText(img, text, (x0+8, y0), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


def fundamental_from_image():
    img1 = cv2.imread('/scratch2/wuti/Others/3DVision/0125-0135/ULSAN HYUNDAI FC vs AL DUHAIL SC CAM1/img/image0001.jpg',0)  #queryimage # left image
    img2 = cv2.imread('/scratch2/wuti/Others/3DVision/0125-0135/ULSAN HYUNDAI FC vs AL DUHAIL SC 16m RIGHT/img/image0001.jpg',0) #trainimage # right image
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    return F


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
    track = np.genfromtxt(args.result_file,delimiter=',',usecols=(1,2,3,4,5)).astype(int)
#    track = np.genfromtxt(args.result_file,delimiter=',',usecols=(1, 2,3,4,5, 7)).astype(int)
    frameidxcol = np.genfromtxt(args.result_file,delimiter=',',usecols=(0)).astype(int)

    # if want to visualize the tracking results of the other camera, also read the results
    if (args.vis_calib_file is not None) and (args.vis_result_file is not None):
        vis_track = np.genfromtxt(args.vis_result_file,delimiter=',',usecols=(1,2,3,4,5)).astype(int)
        vis_frameidxcol = np.genfromtxt(args.vis_result_file,delimiter=',',usecols=(0)).astype(int)
        # load calib and image file list
        vis_calib = np.genfromtxt(args.vis_calib_file,delimiter=',',usecols=(1,2,3,4,5,6))
        vis_imgname = np.genfromtxt(args.vis_calib_file,delimiter=',',usecols=(7), dtype=str)
        vis_framecalib = [int(x.split('.')[0][5:]) for x in vis_imgname]
        # load calib and image file list for the reference camera
        calib = np.genfromtxt(args.calib_file,delimiter=',',usecols=(1,2,3,4,5,6))
        imgname = np.genfromtxt(args.calib_file,delimiter=',',usecols=(7), dtype=str)
        framecalib = [int(x.split('.')[0][5:]) for x in imgname]
    
    # print(vis_framecalib)

    
    # prepare video
    if args.pitchmode:
        output_file = os.path.join(args.result_file[:-len(os.path.basename(args.result_file))], os.path.basename(args.result_file.split('.')[0]+'_3d.mp4'))
    elif args.vis_result_file is not None:
        output_file = os.path.join(os.path.dirname(args.vis_result_file), os.path.basename(args.vis_result_file).split('.')[0]+'_to_'+os.path.basename(args.result_file).split('.')[0]+'_homo.mp4')
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
        # F = fundamental_from_image()
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
            
            # visualize the tracking results of the other camera
            if (args.vis_calib_file is not None) and (args.vis_result_file is not None):
                # get the tracking result of the current frame
                vis_track_cur = vis_track[vis_frameidxcol==frameidx]
                # print(frameidx)
                # print(vis_track_cur)
                # break
                # get the calibration of the two cameras for current frame
                # find index in calib file
                if (not frameidx in framecalib) or (not frameidx in vis_framecalib):
                    continue
                else:
                    # project to soccer pitch
                    calibline = calib[framecalib.index(frameidx)]
                    vis_calibline = vis_calib[vis_framecalib.index(frameidx)]
                    WIDTH, HEIGHT = img.shape[0], img.shape[1]
                    cx,cy = WIDTH/2., HEIGHT/2.

                    # project to soccer pitch
                    # P_ref = computeP(calibline, cx, cy)
                    # P_vis = computeP(vis_calibline, cx, cy)
                    # compute homography
                    # warp_matrix = np.dot(P_ref, np.linalg.pinv(P_vis))
                    
                    # warp_matrix = compute_homo(vis_calibline,calibline,cx,cy)
                    # print(warp_matrix)
                    warp_matrix = homo_from_pose(vis_calibline,calibline,cx,cy)

                    # img2_warp = cv2.warpPerspective(img2, warp_matrix, (HEIGHT, WIDTH))

                    # F = cv2.fundamentalFromProjections(P_ref, P_vis)
                    # F = fundamental_from_pose(calibline, vis_calibline, cx, cy)
                    # print(F)

                    for line in vis_track_cur:
                        if args.xymode:
                            x1, y1, x2, y2 = line[1], line[2], line[3], line[4]
                        else:
                            x1, y1, x2, y2 = line[1], line[2], line[1]+line[3], line[2]+line[4]
                        vis_trace_id = line[0]
                        # compute center point (x as horizontal axis)
                        # cxp, cyp = int((x1+x2)/2), int((y1+y2)/2)
                        cxp, cyp = int((x1+x2)/2), int(y2)
                        # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
                        cp = draw_3d(warp_matrix, cxp, cyp)
                        # cp = warp_pos(vis_calibline, calibline, cx, cy, cxp, cyp)
                        # x = np.vstack((cxp, cyp))
                        # line1 = cv2.computeCorrespondEpilines(x.reshape(-1,1,2), 2,F).reshape(3,1)
                        # line1 = F.dot(homogenize(x))
                        # print(line1.shape)
                        # line1 /= line1[2,0]
                        # draw_line(img, line1,str(vis_trace_id), color_list[vis_trace_id % len(color_list)])
                        cv2.putText(img, str(vis_trace_id), (int(cp[0]), int(cp[1]) - 8), cv2.FONT_HERSHEY_PLAIN, 2, color_list[vis_trace_id % len(color_list)], 2)
                        cv2.circle(img, (int(cp[0]), int(cp[1])), radius=5, color=color_list[vis_trace_id % len(color_list)], thickness=-1)
            videoWriter.write(img)
        cv2.waitKey(0)

if __name__ == '__main__':
    args = config()
    
    main(args)
