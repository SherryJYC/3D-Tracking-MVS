#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:43:40 2021

@author: yujiang
"""
import os 
import json
import argparse
import numpy as np

'''
convert json to txt in format:
    frame id, id, x1, y1, x2, y2, confidence, class (team) id

'''

def loadT(config_file):
    # projection param from proj_config.txt
    param = np.genfromtxt(config_file, delimiter=',').astype(int)
    woffset = param[2]
    hoffset = param[3]
    scale= param[4]/10
    
    T = np.eye(3,3)*scale
    T[0,-1], T[1,-1] = woffset, hoffset
    return T

def json2text(args):
    
    f = open(args.jsonfile,) 
    data = json.load(f)
    prefix = args.jsonfile[:-len(args.jsonfile.split('/')[-1])]
    T = loadT(os.path.join(prefix, 'proj_config.txt'))
    
    playerid = []
    
    cameras = data['cameras']
    for cam in cameras:
        rows = []
        for frame in cam['frames']:
            frameid = frame['frame_id']
            for instance in frame['instances']:
                if instance['id'] not in playerid:
                    playerid.append(instance['id'])
                instance['bbox'] = np.array(instance['bbox']).astype(float)
                xy1 = np.dot(T, np.array([instance['bbox'][0], instance['bbox'][1], 1]))
                xy2 = np.dot(T, np.array([instance['bbox'][2], instance['bbox'][3], 1]))
                
                rows.append([frameid+1, playerid.index(instance['id']), xy1[0], xy1[1], xy2[0], xy2[1], instance['confidence'], instance['class']])
        
        np.savetxt(os.path.join(prefix, cam['camera_id']+'.txt'), np.array(rows).astype('float'), delimiter=',')
        np.savetxt(os.path.join(prefix, 'id_mapping'+'.txt'), np.array(playerid).astype(str), delimiter=',', fmt='%s')
if __name__=="__main__":
    
    a = argparse.ArgumentParser()
    a.add_argument("--jsonfile", help="path to json file", default='./data/0125-0135/0125-0135.json')

    args = a.parse_args()    
    
    json2text(args)
