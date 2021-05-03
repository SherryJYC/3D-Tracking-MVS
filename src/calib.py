import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math

SCALE = 1
plane = np.array([0,1,0])
origin = np.array([0,0,0])
theta = np.linspace(0, 2.*np.pi, 50)
alpha = np.linspace(0.72*np.pi, 1.28*np.pi, 25)
pitch = [
    # home half field
    np.array([[0, 34, 0, 1],
              [0, -34, 0, 1],
              [52.5, -34, 0, 1],
              [52.5, 34, 0, 1],
              [0, 34, 0, 1],]).T,
    
    # home penalty area
    np.array([[52.5, -20.15, 0, 1],
              [52.5,  20.15, 0, 1],
              [36, 20.15, 0, 1],
              [36, -20.15, 0, 1],
              [52.5, -20.15, 0, 1],]).T,
    
    # home goal area
    np.array([[52.5, -9.15, 0, 1],
              [52.5,  9.15, 0, 1],
              [47, 9.15, 0, 1],
              [47, -9.15, 0, 1],
              [52.5, -9.15, 0, 1],]).T,

    # away half field
    np.array([[0, 34, 0, 1],
              [0, -34, 0, 1],
              [-52.5, -34, 0, 1],
              [-52.5, 34, 0, 1],
              [0, 34, 0, 1],]).T,
    
    # away penalty area
    np.array([[-52.5, -20.15, 0, 1],
              [-52.5,  20.15, 0, 1],
              [-36, 20.15, 0, 1],
              [-36, -20.15, 0, 1],
              [-52.5, -20.15, 0, 1],]).T,
    
    # away goal area
    np.array([[-52.5, -9.15, 0, 1],
              [-52.5,  9.15, 0, 1],
              [-47, 9.15, 0, 1],
              [-47, -9.15, 0, 1],
              [-52.5, -9.15, 0, 1],]).T,
    
    # center circle
    np.stack([9.15 * np.cos(theta), 9.15 * np.sin(theta), np.zeros_like(theta), np.ones_like(theta)]),

    # home circle
    np.stack([41.5 + 9.15 * np.cos(alpha), 9.15 * np.sin(alpha), np.zeros_like(alpha), np.ones_like(alpha)]),
    
    # away circle
    np.stack([-41.5 - 9.15 * np.cos(alpha), 9.15 * np.sin(alpha), np.zeros_like(alpha), np.ones_like(alpha)]),
]

def project(pts_3d, P):
    projected = P @ pts_3d
    projected /= projected[-1]
    x, y = projected[:2]
    return x, y
    
def display_soccer_pitch(img, P):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 9))
    plt.imshow(img)

    for pts in pitch:
        pts[[1,2]] = pts[[2,1]]
        x, y = project(pts, P)
        plt.plot(x, y, 'r-')
    
    plt.show()
   
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

def cross(e):
    return np.array([[0,e[2],-e[1]],[-e[2],0,e[0]],[e[1],-e[0],0]])

def skew(w):
    top = np.hstack([-w[3]*np.diag([1,1,1]), cross(w[:3])])
    bottom = np.hstack([w[:3],np.zeros(3)])

    return np.vstack([top,bottom])

def point_normal_eq(normal, pt):
    return np.hstack([normal,-np.inner(normal,pt)]).reshape(-1,1)

def backproject_pitch(P, x, C_cam):
    X = np.dot(np.linalg.pinv(P),x)
    X /= X[-1]

    C_homo = np.hstack([C_cam, 1]).reshape(-1,1)
    p = np.dot(X.T,plane_normal)*C_homo-np.dot(C_homo.T,plane_normal)*X

    p /= p[-1]
    return p[0],p[2]

def display_soccer_pitch_ground(points,imgname,C_cam):
    plt.figure(figsize=(16, 9))

    for pts in pitch:
        pts[[1,2]] = pts[[2,1]]
        px, py = pts[0],pts[2]
        plt.plot(SCALE*px, SCALE*py, 'r-')
    
    plt.scatter(C_cam[0],-C_cam[2])
    plt.annotate('Camera',(C_cam[0],-C_cam[2]))

    for p in points:
        plt.scatter(p[0],-p[1])
        plt.annotate(p[2],(p[0],-p[1]))
    
    # plt.show()
    plt.savefig(imgname)
        


if __name__ == "__main__":

    img = cv2.imread("/scratch2/wuti/Others/3DVision/0125-0135/ULSAN HYUNDAI FC vs AL DUHAIL SC 16m CAM1/img/image0001.jpg")
    calib_file = '/scratch2/wuti/Others/3DVision/calibration_results/0125-0135/CAM1/calib.txt'
    calib = np.genfromtxt(calib_file,delimiter=',',usecols=(1,2,3,4,5,6))
    imgname = np.genfromtxt(calib_file,delimiter=',',usecols=(7),dtype=str)
    framecalib = [int(x.split('.')[0][5:]) for x in imgname]
    result_file = '/scratch2/wuti/Others/3DVision/cam1_result_filtered/cam1_filtered_4.24.txt'
    xymode = False
    tracks = np.genfromtxt(result_file,delimiter=',',usecols=(0,1,2,3,4,5)).astype(int)

    print(imgname)
    
    cx,cy = 960, 540
    Projections = [computeP(calibline, cx, cy) for calibline in calib]
    print(len(Projections),len(calib))

    # lines = tracks[tracks[:,0]==1]
    # print(lines)
    # P = computeP(calib[0], cx, cy)
    C_cam = calib[0,-3:]
    # print(C_cam)

    plane_normal = point_normal_eq(plane,origin)
    # print(plane_normal)
    # points = []
    # for line in lines:
    #     if xymode:
    #         x1, y1, x2, y2 = line[2], line[3], line[4], line[5]
    #     else:
    #         x1, y1, x2, y2 = line[2], line[3], line[2]+line[4], line[3]+line[5]
    #     tx, ty = backproject_pitch(P,np.array([(x1+x2)/2,y2,1]).reshape(-1,1),C_cam)

    #     points.append([tx, ty, line[1]])
    # display_soccer_pitch_ground(points, '16m_right.png',C_cam)
    
    # save tracking results coordinates on the pitch
    tracks_pitch = []

    # print(Projections)
    for track in tracks:
        if xymode:
            x1, y1, x2, y2 = track[2], track[3], track[4], track[5]
        else:
            x1, y1, x2, y2 = track[2], track[3], track[2]+track[4], track[3]+track[5]
        # get the calibration of the corresponding frames
        # print(framecalib == track[0])
        if track[0] not in framecalib:
            continue
        P = Projections[framecalib.index(track[0])]
        C_cam = calib[framecalib.index(track[0]),-3:]

        tx, ty = backproject_pitch(P,np.array([(x1+x2)/2,y2,1]).reshape(-1,1),C_cam)
        
        # frame id, track id, x, z
        tracks_pitch.append([track[0], track[1], tx, ty])
    
    np.savetxt('/scratch2/wuti/Others/3DVision/cam1_result_filtered/cam1_filtered_4.24_pitch.txt', np.array(tracks_pitch), delimiter=',')

    # img2 = cv2.imread("/scratch2/wuti/Others/3DVision/0125-0135/ULSAN HYUNDAI FC vs AL DUHAIL SC CAM1/img/image0001.jpg")
    # calib_file2 = '/scratch2/wuti/Others/3DVision/calibration_results/0125-0135/CAM1/calib.txt'
    # calib2 = np.genfromtxt(calib_file2,delimiter=',',usecols=(1,2,3,4,5,6))
    # result_file2 = '/scratch2/wuti/Others/3DVision/cam1_result/cam1.txt'
    # track2 = np.genfromtxt(result_file2,delimiter=',',usecols=(0,1,2,3,4,5)).astype(int)
    # lines2 = track2[track2[:,0]==1]
    # P2 = computeP(calib2[0], cx, cy)
    # C_cam2 = calib2[0,-3:]
    # points2 = []

    # for line2 in lines2:
    #     x1, y1, x2, y2 = line2[2], line2[3], line2[4], line2[5]
    #     tx, ty = backproject_pitch(P2,np.array([(x1+x2)/2,y2,1]).reshape(-1,1),C_cam2)

    #     points2.append([tx, ty, line2[1]])

    # display_soccer_pitch_ground(points2, 'cam1.png', C_cam2)

#    K = np.asarray([[ 5502.18, 0.0,     2048.0],
#                    [ 0.0,     5502.18, 1152.0],
#                    [ 0.0,     0.0,     1.0]])
#    R = np.asarray([[ 0.08008822, -0.99655787, -0.02140748],
#                    [-0.41593093, -0.01389338, -0.90929007],
#                    [ 0.90586276,  0.08172745, -0.41561194]])
#    t = np.asarray( [[-3.52350603, -2.12259918, 85.24450696]]).T
#    P = K @ np.concatenate([R, t], axis=1)
    
    # display_soccer_pitch(img, P)
