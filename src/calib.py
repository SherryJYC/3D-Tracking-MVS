import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math


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

if __name__ == "__main__":
    img = cv2.imread("./data/0125-0135/CAM1/img/image0125.png")
    calib_file = 'data/calibration_results/0125-0135/CAM1/calib.txt'
    calib = np.genfromtxt(calib_file,delimiter=',',usecols=(1,2,3,4,5,6))
    cx,cy = 960, 540
    P = computeP(calib[125], cx, cy)

#    K = np.asarray([[ 5502.18, 0.0,     2048.0],
#                    [ 0.0,     5502.18, 1152.0],
#                    [ 0.0,     0.0,     1.0]])
#    R = np.asarray([[ 0.08008822, -0.99655787, -0.02140748],
#                    [-0.41593093, -0.01389338, -0.90929007],
#                    [ 0.90586276,  0.08172745, -0.41561194]])
#    t = np.asarray( [[-3.52350603, -2.12259918, 85.24450696]]).T
#    P = K @ np.concatenate([R, t], axis=1)
    
    display_soccer_pitch(img, P)