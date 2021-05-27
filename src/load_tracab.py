import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import math

from calib import computeP

FPS = 25
team_color_list = [(0,0,255), (255,255,255), (0,255,0), (255,0,0)]

def project(pts_3d, P):
    projected = P[:3, :3] @ pts_3d + P[:3, 3, None]
    projected /= projected[-1]
    x, y = projected[:2]
    return x, y

def draw_tracab(pts_3d, P):
    xs, ys = project(pts_3d, P)
    plt.plot(xs, ys, 'r+')

def load_tracab(filename, length):
    with open(filename) as f:
        data = f.read().splitlines()[:length]
    for line in data:
        line = line.strip(':').split(':')[1]
        players = filter(lambda x: int(x[0]) in [0, 1], line.strip(';').split(';'))
        track = np.asarray(list(map(lambda x: x.strip(',').split(',')[:5]+[0], players))).astype(float)
        # print(track.shape)
        # team_id, player_id, jersey_number, X, Y, Z
        track[:,3:5] /= -100
        track[:,3] = -track[:,3]
        # swap the y and z coordinate to keep the consistency with the others
        track[:,[4,5]] = track[:,[5,4]]
        yield track

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="image to be calibrated")
    parser.add_argument("--calib_path", type=str)
    parser.add_argument("--gt_path", type=str)
    args = parser.parse_args()

    # img_file = args.image
    imgs = glob(os.path.join(args.img_path, "*"))
    print(imgs)

    H, W, _ = cv2.imread(imgs[0]).shape

    cx, cy = W/2, H/2

    calib = np.genfromtxt(args.calib_path, delimiter=',', usecols=(1, 2, 3, 4, 5, 6))
    imgname = np.genfromtxt(args.calib_path, delimiter=',', usecols=(7), dtype=str)
    framecalib = [x.split('.')[0] for x in imgname]
    print(framecalib)
    Projections = [computeP(calibline, cx, cy) for calibline in calib]
    
    videoWriter = cv2.VideoWriter(args.gt_path.replace('.txt','.mp4'), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, (W, H))

    tracks = []
    i = 0
    for track in load_tracab(args.gt_path, 251):
        # read the corresponding image
        print("write %s" %os.path.basename(imgs[i]))
        img = cv2.imread(imgs[i])
        name = os.path.basename(imgs[i]).split('.')[0]
        i+= 1
        if name not in framecalib:
            continue
        P = Projections[framecalib.index(name)]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for player in track:
            # parse track
            pos = player[3:].reshape(-1,1)
            xs, ys = project(pos, P)
            cv2.circle(img, (int(xs), int(ys)), radius=5,color=team_color_list[int(player[0])],thickness=-1)
            cv2.putText(img, str(int(player[1]))+'('+str(int(player[2]))+')', (int(xs), int(ys) - 8), cv2.FONT_HERSHEY_PLAIN, 1,team_color_list[int(player[0])], 2)
        videoWriter.write(img)

        # append track to tracks list
        tracks.append(np.hstack([i*np.ones((track.shape[0],1)),track]))
    
    # frame_id, team_id, player_id, jersey_number, X, Y, Z
    tracks = np.vstack(tracks)

    # get the positions on pitch coordinates
    # frame_id, jersey_number, X, Z, team_id, player_id
    tracks_on_pitch = tracks[:,[0,3,4,6,1,2]]
    print(tracks_on_pitch)
    np.savetxt(os.path.join(os.path.dirname(args.gt_path),'gt_pitch.txt'),tracks_on_pitch,delimiter=',')
