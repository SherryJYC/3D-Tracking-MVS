#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:43:40 2021

@author: yujiang
"""
import os 
import json
import numpy as np
import argparse

def json2text(jsonfile):
    f = open(jsonfile,) 
    data = json.load(f)
    
    rows = []
    objectkey = np.empty((70,))
    for obj in data['objects']:
        # get drone id: 1, 2, 3 => 0, 1, 2
        # bbox -> 1-30
        # goalkeep -> 31-32
        # coach -> 41
        # judge -> 51+
        print(obj['classTitle'])
        droneidx = 60
        if obj['classTitle'][0] == 'p':
            # if obj['classTitle'] == 'p':
            #     droneidx = 1
            # else:
            #     droneidx = int(obj['classTitle'][1:])+1

            droneidx = int(obj['classTitle'][1:])
        if 'goalkeeper' in obj['classTitle']:
            if obj['classTitle'] == 'goalkeeper':
                droneidx = 31
            else:
                droneidx = int(obj['classTitle'][10:])+31
        if 'coach' in obj['classTitle']:
            if obj['classTitle'] == 'coach':
                droneidx = 41
            else:
                droneidx = int(obj['classTitle'][5:])+41
        if 'referee' in obj['classTitle']:
            if obj['classTitle'] == 'referee':
                droneidx = 51
            else:
                droneidx = int(obj['classTitle'][7:])+51
        # droneidx = int(obj['classTitle'][-1]) - 1
        print(obj)
        objectkey[droneidx] = obj['key'] #obj['id']

    objectkey = objectkey.astype('int32').tolist()
    print(objectkey)

    for frame in data['frames']:
        frameid = frame['index']
        instances = frame['figures']
        for instance in instances:
            points = instance['geometry']['points']['exterior']
            w = points[1][0]-points[0][0]
            h = points[1][1]-points[0][1]
            objectid = objectkey.index(instance['objectId'])
            rows.append([frameid, objectid, points[0][0], points[0][1], w, h, points[1][0], points[1][1]])
    return rows

def muljson2text(json_folder):
    json_files = [file for file in os.listdir(json_folder) if os.path.splitext(file)[-1]=='.json']
    start = 0
    rows = []
    json_files.sort()
    print(json_files)
    for json_file in json_files:
        f = open(os.path.join(json_folder, json_file),) 
        data = json.load(f)
        framecount = data['framesCount']
        
        # for a single json, convert to txt
        objectkey = np.empty((3,))
        for obj in data['objects']:
            # get drone id: 1, 2, 3 => 0, 1, 2
            droneidx = int(obj['classTitle'][-1]) - 1
            objectkey[droneidx] = obj['id']

        objectkey = objectkey.astype('int32').tolist()
    
        for frame in data['frames']:
            frameid = frame['index']
            instances = frame['figures']
            for instance in instances:
                points = instance['geometry']['points']['exterior']
                print(objectkey)
                w = points[1][0]-points[0][0]
                h = points[1][1]-points[0][1]
                objectid = objectkey.index(instance['objectId'])
                rows.append([frameid+start, objectid, points[0][0], points[0][1], w, h, points[1][0], points[1][1]])
                
        # accumulate start frame idx
        start +=framecount
    return rows

if __name__=="__main__":
    
    a = argparse.ArgumentParser()
    a.add_argument("--jsonfile", help="path to json file", default='./data/0125-0135/RIGHT/right_gt.json')
    a.add_argument("--jsonfolder", help="path to json folder", default='./data/dataset5/cam4/json')
    a.add_argument("--output", help="path to output text file", default='./data/0125-0135/RIGHT/right_gt.txt')
    a.add_argument("--muljson", help="whether to combine multiple json file", default=True)
    args = a.parse_args()    
    
    # if args.muljson:
    #     print('mul json2text')
    #     np.savetxt(args.output, np.array(muljson2text(args.jsonfolder)), delimiter=',')
    # else:
    print('json2text')
    np.savetxt(args.output, np.array(json2text(args.jsonfile)), delimiter=',')
