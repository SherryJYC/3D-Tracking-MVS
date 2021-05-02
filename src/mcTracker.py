#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 2 16:43:40 2021

@author: yujiang
"""

"""
combine two cameras from tracking result <frame id, obj id, x, z>

class file
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

class Position():
    """
    store a position object (3d)
    """
    def __init__(self, x, z, t, cam=None, objid=None):
        self.x = x
        self.z = z
        self.t = t
        self.info = {} # {'cams':, 'ids':}


class Target():
    """
    store a target object
    """
    def __init__(self, pos, trackid):
        self.id = trackid
        self.last_pos = pos
        self.last_frame = pos.t
        self.pos_num = 1
        self.pos_list = [pos]

    def add_pos(self, pos):
        '''
        add newly tracked bbox
        '''
        self.pos_list.append(pos)
        self.pos_num+=1
        self.last_pos = pos
        self.last_frame = pos.t

class Camera():
    """
    store detection loaded from cam file
    """
    def __init__(self, input_file, camid):
        self.cam_file = input_file
        self.bboxs = None
        self.width = 55
        self.height = 36
        self.camid = camid
        self.tstart = 1
        self.tend = -1

        self.read_pos_from_file()

    def read_pos_from_file(self):
        # <frame id, obj id, x, z>
        newbboxs = []
        bboxs = np.genfromtxt(self.cam_file, delimiter=',', usecols=(0, 1, 2, 3))
        for box in bboxs:
            t, objid, x, z = box[0], box[1], box[2], box[3]
            # check if position valid (inside)
            if abs(x) > self.width or abs(z) > self.height:
                continue
            # define region
            regionid = 0
            if x > 0 and x <= self.width:
                if z > 0 and z <= self.height:
                    regionid = 1
                else:
                    regionid = 2
            else:
                if z > 0 and z <= self.height:
                    regionid = 3
                else:
                    regionid = 4

            newbboxs.append([t, objid, x, z, regionid])
            self.tend = t
        self.bboxs = np.array(newbboxs)

class Pitch():
    """
    store and update 3d targets (tracked)
    """
    def __init__(self):
        self.tstart = 1
        self.tend = -1
        self.cam_list = [] # only accept 2 cams now
        self.target_list = []
        self.trash_score = 5
        self.free_trackid = 1 # min availabel track id

    def add_cam(self, cam):
        self.cam_list.append(cam)
        if self.tend < cam.tend:
            self.tend = cam.tend
        if self.tstart > cam.tstart:
            self.tstart = cam.tstart

    def initTarget(self):
        '''
        use bbox from cameras in 1st frame to init targets
        :return: target list
        '''
        bbox1 = self.cam_list[0].bboxs
        bbox1 = bbox1[bbox1[:, 0]==1]
        bbox2 = self.cam_list[1].bboxs
        bbox2 = bbox2[bbox2[:, 0]==1]

        self.matchInRegion(region=1, bbox1=bbox1, bbox2=bbox2, tcur=1)
        self.matchInRegion(region=2, bbox1=bbox1, bbox2=bbox2, tcur=1)
        self.matchInRegion(region=3, bbox1=bbox1, bbox2=bbox2, tcur=1)
        self.matchInRegion(region=4, bbox1=bbox1, bbox2=bbox2, tcur=1)

        print('initialize with {} targets'.format(self.free_trackid))

    def trackTarget(self):
        for t in range(self.tstart, self.tend):
            pass

    def matchInRegion(self, region, bbox1, bbox2, tcur):
        bbox1_region = bbox1[bbox1[:, -1] == region]
        bbox1_region = bbox1_region[:, 2:4]
        bbox2_region = bbox2[bbox2[:, -1] == region]
        bbox2_region = bbox2_region[:, 2:4]

        # compute cost matrix
        costmat = np.full((len(bbox1_region), len(bbox2_region)), np.inf)
        for i, xz1 in enumerate(bbox1_region):
            for j, xz2 in enumerate(bbox2_region):
                costmat[i, j] = np.linalg.norm(xz1 - xz2)
        # add trash bin
        costmat = np.vstack((costmat, np.ones((len(bbox1_region), len(bbox2_region))) * self.trash_score))
        costmat = np.hstack((costmat, np.ones((len(bbox1_region) * 2, len(bbox2_region))) * self.trash_score))

        row_ind, col_ind = linear_sum_assignment(costmat)

        # form targets in matched pair
        valid_row, valid_col = [], []
        for r, c in zip(row_ind, col_ind):
            if r >= len(bbox1_region) or c >= len(bbox2_region):
                continue
            valid_row.append(r)
            valid_col.append(c)
            # create target
            x, z, t = (bbox1_region[r, 0] + bbox2_region[c, 0]) / 2, (bbox1_region[r, 1] + bbox2_region[c, 1]) / 2, 1
            target = Target(Position(x, z, t), self.free_trackid)
            self.target_list.append(target)
            self.free_trackid += 1

        # form targets in unmatched pair in cam1
        for r in range(len(bbox1_region)):
            if r in valid_row:
                continue
            x, z, t = bbox1_region[r, 0], bbox1_region[r, 1], tcur
            target = Target(Position(x, z, t), self.free_trackid)
            self.target_list.append(target)
            self.free_trackid += 1

        # form targets in unmatched pair in cam2
        for c in range(len(bbox2_region)):
            if c in valid_col:
                continue
            x, z, t = bbox2_region[c, 0], bbox2_region[c, 1], tcur
            target = Target(Position(x, z, t), self.free_trackid)
            self.target_list.append(target)
            self.free_trackid += 1