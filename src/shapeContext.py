#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:38:49 2021

@author: yujiang
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


nbBins_theta = 12
nbBins_r = 5
smallest_r = 1/8 #length of the smallest radius (assuming normalized distances)
biggest_r = 3 #5, length of the biggest radius (assuming normalized distances)
maxIter = 1


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, rho
    
def sc_compute(X,nbBins_theta,nbBins_r,smallest_r,biggest_r):
    '''
    compute shape descriptor based on:
            Belongie, S., Malik, J., & Puzicha, J. (2002). 
            Shape matching and object recognition using shape contexts.
            
            @param:
            set of points (nparray), X (nx2)
            number of bins in the angular dimension, nbBins_theta 
            number of bins in the radial dimension, nbBins_r
            the length of the smallest radius, smallest_r
            the length of the biggest radius, biggest_r                    
    '''
    # total num of points
    n = X.shape[0]
    descriptor = defaultdict(int)
    
    # def log-polar coord system
    minr = np.log(smallest_r);
    maxr = np.log(biggest_r);
    dr = (maxr-minr)/nbBins_r; # equally radius interval
    r = np.linspace(minr,maxr-dr,nbBins_r+1); # get all possible radius
    
    # build polar grid in angular dimenstion
    dtheta = 2*np.pi/nbBins_theta; # equally angle interval
    theta = np.linspace(0,2*np.pi-dtheta,nbBins_theta+1)

    norm_scale = np.mean(cdist(X,X))
    
    # for each point, compute descriptor
    for i in range(n):
        temp_pt = X[i,:]
        # calculate distance from temp point to all other points in X
        dist = temp_pt - X 
        dist = np.delete(dist, i, 0) # no need to compute dist for self-self pair
        # convert into polar coord
        dist_theta, dist_r = cart2pol(dist[:,0], dist[:,1])
        dist_r = dist_r / norm_scale # normalization
        histbins, _, _ = np.histogram2d(dist_theta, np.log(dist_r), bins=(theta, r))
        histbins[histbins==0.] = np.finfo(float).eps
        descriptor[i] = histbins
        
    return descriptor

def chi2_cost(s1, s2):
    '''
    compute distance between 2 shape descriptors (use X^2)
    '''
    m = len(s1)
    n = len(s2)
    costmat = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            cost = 0.5*np.divide(np.square(s1[i]-s2[j]), (s1[i]+s2[j]))
            cost[np.isnan(cost)] = 0
            costmat[i, j] = np.sum(cost)
    
    return costmat

def tps_model(X,Y,lam):
    '''
    compute transformation based on thin plate spline model
    return weights and cost/energy
    @param:
        X, Y: corresponded points
        lam: regularisation param
        w_x: param in TPS model along x
        w_y: param in TPS model along y
        E: total bending energy
    '''
    n = X.shape[0]
    
    K = cdist(X, X) # nxn
    K = np.multiply(np.square(K), np.log(np.square(K)))
    K[np.isnan(K)] = 0

    #  compute P (the i-th row of P is (1,xi,yi)), nx3
    P = np.hstack((np.ones((n,1)), X))
    
    # solve linear system Ax = b
    # A = [K+lam*I  P]
    #     [P.T      0]
    A = np.vstack((np.hstack((K+lam*np.identity(n), P)), np.hstack((P.T, np.zeros((3,3)))))) # (nx3)x(nx3)
    bx = np.vstack((Y[:,0].reshape((-1, 1)), np.zeros((3,1)))) # (n+3) x 1
    by = np.vstack((Y[:,1].reshape((-1, 1)), np.zeros((3,1))))
    
    # x = [w | a]
    x_x = np.linalg.pinv(A).dot(bx)
    x_y = np.linalg.pinv(A).dot(by)
    
    w_x = x_x[:n]
    w_y = x_y[:n]
        
    # compute energy/cost
    E = w_x.T.dot(K).dot(w_x) + w_y.T.dot(K).dot(w_y)

    return x_x,x_y,E

def shapeMatch(X, Y):
    '''
    Compute distance between 2 tracks
    
    1. compute shape descriptor for each track
    2. compute TPS transformation between them
    3. compute dist = dist(p - T(q)) + dist(q - T(p))
    
    return bending energy
    '''
    curX = X # nx2
    curIter = 1;
    nbSamples = X.shape[0]
    
    # iterations
    while (curIter<=maxIter):
       print('current iteration: '+str(curIter))

       # compute descriptor
       print('computing shape contexts')
       ShapeDescriptors1 = sc_compute(curX,nbBins_theta,nbBins_r,smallest_r,biggest_r)
       ShapeDescriptors2 = sc_compute(Y,nbBins_theta,nbBins_r,smallest_r,biggest_r)
       
       # set lambda here      
       mean_dist = np.mean(cdist(Y,Y))
       lam = mean_dist**2 
       
       print('compute cost')
       # compute dist between descriptor
       costmat = chi2_cost(ShapeDescriptors1,ShapeDescriptors2)

       # find correspondence with hungarian algorithm
       row_idx, col_idx = linear_sum_assignment(costmat)
       
#       Xwarped = curX[row_idx,:]
       Xunwarped = X[row_idx,:]
       
       print('compute transformation')
       # compute transformation based on tps (w_x: nx1)
       w_x, w_y, E = tps_model(Xunwarped, Y[col_idx, :],lam)
       
       print('current energy: '+str(E))
       
       if E < 1e-5:
           E = 0
       # wrap coord
       # x[n:n+3].t (1x3) dot [ones; X.t] (3xn) => 1xn

       fx_aff = w_x[nbSamples:nbSamples+3].T.dot(np.vstack((np.ones((1, nbSamples)),X.T)))        
       d2 = np.square(cdist(Xunwarped, X)) # nxn
       d2[d2<0] = 0
       U = np.multiply(d2, np.log(d2+np.finfo(float).eps)) # nxn
       fx_wrp = w_x[:nbSamples].T.dot(U) # 1xn
       fx = fx_aff+fx_wrp # 1xn       
       fy_aff = w_y[nbSamples:nbSamples+3].T.dot(np.vstack((np.ones((1, nbSamples)),X.T))) 
       fy_wrp = w_y[:nbSamples].T.dot(U) # 1xn
       fy = fy_aff+fy_wrp

       curX = np.vstack((fx, fy)).T     
       curIter+=1
       
    return E

        
