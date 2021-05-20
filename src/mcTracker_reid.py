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

PENALTY_REID = 3
TRASH_SCORE = 10


class Position():
    """
    store a position object (3d)
    """
    def __init__(self, x, z, t, objid=None, cam=None):
        self.x = x
        self.z = z
        self.t = int(t)
        self.info = {}  # {'cams':, 'ids':}
        self.objid = objid


class Target():
    """
    store a target object
    """
    def __init__(self, pos, trackid, teamid):
        self.id = trackid
        self.last_pos = pos
        self.last_frame = pos.t
        self.pos_num = 1
        self.pos_list = [pos]
        self.v = [0, 0]

        self.teamid = teamid

    def add_pos(self, pos):
        '''
        add newly tracked bbox
        and update velocity 
        '''
        self.pos_list.append(pos)
        self.pos_num += 1
        self.last_pos = pos
        self.last_frame = pos.t
        self.update_vel()

    def get_last_pos(self, cur_frame):

        if self.v[0] == 0 and self.v[1] == 0:
            return self.last_pos
        else:
            x = (cur_frame - self.last_pos.t) * self.v[0] + self.last_pos.x
            z = (cur_frame - self.last_pos.t) * self.v[1] + self.last_pos.z
            return Position(x, z, self.last_pos.t, self.last_pos.objid)

    def set_id(self, newid):
        self.id = newid

    def update_vel(self):
        if (self.pos_num < 2):
            return
        elif (self.pos_num < 6):
            vx = (self.pos_list[self.pos_num - 1].x -
                  self.pos_list[self.pos_num - 2].x) / (
                      self.pos_list[self.pos_num - 1].t -
                      self.pos_list[self.pos_num - 2].t)
            vy = (self.pos_list[self.pos_num - 1].z -
                  self.pos_list[self.pos_num - 2].z) / (
                      self.pos_list[self.pos_num - 1].t -
                      self.pos_list[self.pos_num - 2].t)
            self.v = [vx, vy]
        else:
            vx1 = (self.pos_list[self.pos_num - 1].x -
                   self.pos_list[self.pos_num - 4].x) / (
                       self.pos_list[self.pos_num - 1].t -
                       self.pos_list[self.pos_num - 4].t)
            vy1 = (self.pos_list[self.pos_num - 1].z -
                   self.pos_list[self.pos_num - 4].z) / (
                       self.pos_list[self.pos_num - 1].t -
                       self.pos_list[self.pos_num - 4].t)
            vx2 = (self.pos_list[self.pos_num - 2].x -
                   self.pos_list[self.pos_num - 5].x) / (
                       self.pos_list[self.pos_num - 2].t -
                       self.pos_list[self.pos_num - 5].t)
            vy2 = (self.pos_list[self.pos_num - 2].z -
                   self.pos_list[self.pos_num - 5].z) / (
                       self.pos_list[self.pos_num - 2].t -
                       self.pos_list[self.pos_num - 5].t)
            vx3 = (self.pos_list[self.pos_num - 3].x -
                   self.pos_list[self.pos_num - 6].x) / (
                       self.pos_list[self.pos_num - 3].t -
                       self.pos_list[self.pos_num - 6].t)
            vy3 = (self.pos_list[self.pos_num - 3].z -
                   self.pos_list[self.pos_num - 6].z) / (
                       self.pos_list[self.pos_num - 3].t -
                       self.pos_list[self.pos_num - 6].t)
            vx, vy = (vx1 + vx2 + vx3) / 3, (vy1 + vy2 + vy3) / 3
            self.v = [vx, vy]


class Camera_reid():
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
        # <frame id, obj id, x, z> =>  <frame id, obj id, x, z, regionid, teamid>
        newbboxs = []
        bboxs = np.genfromtxt(self.cam_file,
                              delimiter=',',
                              usecols=(0, 1, 2, 3, 4))
        for box in bboxs:
            t, objid, x, z, teamid = box[0], box[1], box[2], box[3], box[4]
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

            newbboxs.append([t, objid, x, z, regionid, teamid])
            self.tend = t
        self.bboxs = np.array(newbboxs)


class Pitch_reid():
    """
    store and update 3d targets (tracked)
    """
    def __init__(self, output=None):
        self.tstart = 1
        self.tend = -1
        self.cam_list = []  # only accept 2 cams now
        self.target_list = []
        self.trash_score = TRASH_SCORE
        self.free_trackid = 1  # min availabel track id
        self.temp_target_list = []

        self.disappear_allow = 5

        self.output = output

    def __call__(self):
        '''
        main process
        '''
        self.initTarget()
        self.trackTarget()
        self.saveResult()

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
        bbox1 = bbox1[bbox1[:, 0] == self.tstart]
        bbox2 = self.cam_list[1].bboxs
        bbox2 = bbox2[bbox2[:, 0] == self.tstart]

        self.matchInRegion(region=1,
                           bbox1=bbox1,
                           bbox2=bbox2,
                           tcur=self.tstart)
        self.matchInRegion(region=2,
                           bbox1=bbox1,
                           bbox2=bbox2,
                           tcur=self.tstart)
        self.matchInRegion(region=3,
                           bbox1=bbox1,
                           bbox2=bbox2,
                           tcur=self.tstart)
        self.matchInRegion(region=4,
                           bbox1=bbox1,
                           bbox2=bbox2,
                           tcur=self.tstart)

        print('initialize with {} targets'.format(self.free_trackid))

    def trackTarget(self):
        self.tstart, self.tend = int(self.tstart), int(self.tend)
        for t in range(self.tstart + 1, self.tend):
            # check each camera (tracklet-to-tracklet)
            bbox1 = self.cam_list[0].bboxs
            bbox1 = bbox1[bbox1[:, 0] == t]
            bbox2 = self.cam_list[1].bboxs
            bbox2 = bbox2[bbox2[:, 0] == t]

            # empty temp target list
            self.temp_target_list = []

            self.matchInRegion(region=1,
                               bbox1=bbox1,
                               bbox2=bbox2,
                               tcur=t,
                               temp=True)
            self.matchInRegion(region=2,
                               bbox1=bbox1,
                               bbox2=bbox2,
                               tcur=t,
                               temp=True)
            self.matchInRegion(region=3,
                               bbox1=bbox1,
                               bbox2=bbox2,
                               tcur=t,
                               temp=True)
            self.matchInRegion(region=4,
                               bbox1=bbox1,
                               bbox2=bbox2,
                               tcur=t,
                               temp=True)

            # compare newly detected tracks to exisint target (tracklet-to-target)
            # combine temp_target_list to target_list
            costmat = np.full(
                (len(self.target_list), len(self.temp_target_list)), np.inf)
            for i, pos in enumerate(self.target_list):
                target_last_pos = pos.get_last_pos(t)
                x, z, t_pos, teamid = target_last_pos.x, target_last_pos.z, target_last_pos.t, pos.teamid
                #                x, z, t_pos = pos.last_pos.x, pos.last_pos.z, pos.last_pos.t

                for j, pos_temp in enumerate(self.temp_target_list):
                    x_temp, z_temp, t_pos_temp, teamid_temp = pos_temp.last_pos.x, pos_temp.last_pos.z, pos_temp.last_pos.t, pos_temp.teamid
                    # inf cost for objects far away temporally
                    if abs(t_pos - t_pos_temp) > self.disappear_allow:
                        continue
                    # more cost for objects with different team id
                    costmat[i, j] = np.linalg.norm(
                        np.array([x, z] - np.array([x_temp, z_temp])))
                    if not teamid == teamid_temp:
                        costmat[i, j] *= PENALTY_REID
            # add trash bin
            costmat = np.vstack(
                (costmat,
                 np.ones((len(self.target_list), len(self.temp_target_list))) *
                 self.trash_score))
            costmat = np.hstack(
                (costmat,
                 np.ones(
                     (len(self.target_list) * 2, len(self.temp_target_list))) *
                 self.trash_score))

            row_ind, col_ind = linear_sum_assignment(costmat)

            # form targets in matched pair
            valid_col = []
            for r, c in zip(row_ind, col_ind):
                if r >= len(self.target_list) or c >= len(
                        self.temp_target_list):
                    continue
                valid_col.append(c)

                # update target
                newpos = self.temp_target_list[c].get_last_pos(t)
                self.target_list[r].add_pos(newpos)
                # also update velocity for target

            # for target in unmatched, but newly detected track (col)
            for c in range(len(self.temp_target_list)):
                if c not in valid_col:
                    target = self.temp_target_list[c]
                    target.set_id(self.free_trackid)
                    self.target_list.append(target)
                    self.free_trackid += 1

        print('tracked {} targets at end frame {}'.format(
            len(self.target_list), self.tend))

    def matchInRegion(self, region, bbox1, bbox2, tcur, temp=False):
        bbox1_region_full = bbox1[bbox1[:, -2] == region]
        bbox1_region = bbox1_region_full[:, 2:4]
        bbox2_region_full = bbox2[bbox2[:, -2] == region]
        bbox2_region = bbox2_region_full[:, 2:4]

        # compute cost matrix
        costmat = np.full((len(bbox1_region), len(bbox2_region)), np.inf)
        for i, xz1 in enumerate(bbox1_region):
            for j, xz2 in enumerate(bbox2_region):

                costmat[i, j] = np.linalg.norm(xz1 - xz2)
                # check team id, assign more dist for different team id
                if not bbox1_region_full[i, -1] == bbox2_region_full[j, -1]:
                    costmat[i, j] *= PENALTY_REID

        # add trash bin
        costmat = np.vstack(
            (costmat, np.ones(
                (len(bbox1_region), len(bbox2_region))) * self.trash_score))
        costmat = np.hstack(
            (costmat, np.ones((len(bbox1_region) * 2, len(bbox2_region))) *
             self.trash_score))

        row_ind, col_ind = linear_sum_assignment(costmat)

        # form targets in matched pair
        valid_row, valid_col = [], []
        for r, c in zip(row_ind, col_ind):
            if r >= len(bbox1_region) or c >= len(bbox2_region):
                continue
            valid_row.append(r)
            valid_col.append(c)
            # create target
            x, z, t, teamid, objid = (bbox1_region[r, 0] + bbox2_region[c, 0]) / 2, (
                bbox1_region[r, 1] +
                bbox2_region[c, 1]) / 2, tcur, bbox1_region_full[r, -1], bbox1_region_full[r, 1]
            target = Target(Position(x, z, t, objid), self.free_trackid, teamid)

            if temp:
                self.temp_target_list.append(target)
            else:
                self.target_list.append(target)
                self.free_trackid += 1

        # form targets in unmatched pair in cam1
        for r in range(len(bbox1_region)):
            if r in valid_row:
                continue
            x, z, t, teamid, objid = bbox1_region[r, 0], bbox1_region[
                r, 1], tcur, bbox1_region_full[r, -1], bbox1_region_full[r, 1]
            target = Target(Position(x, z, t, objid), self.free_trackid, teamid)

            if temp:
                self.temp_target_list.append(target)
            else:
                self.target_list.append(target)
                self.free_trackid += 1

        # form targets in unmatched pair in cam2
        for c in range(len(bbox2_region)):
            if c in valid_col:
                continue
            x, z, t, teamid, objid = bbox2_region[c, 0], bbox2_region[
                c, 1], tcur, bbox2_region_full[c, -1], bbox2_region_full[c, 1]
            target = Target(Position(x, z, t, objid), self.free_trackid, teamid)

            if temp:
                self.temp_target_list.append(target)
            else:
                self.target_list.append(target)
                self.free_trackid += 1

    def saveResult(self):
        '''
        save track to result
        '''
        if self.output is None:
            return

        result = []
        for target in self.target_list:
            for pos in target.pos_list:
                result.append([pos.t, target.id, pos.x, pos.z, target.teamid, pos.objid])
        result = np.array(result)

        # sort by frameid
        result = result[result[:, 0].argsort()]

        np.savetxt(self.output, result, delimiter=',')
        print('save to ' + self.output)
