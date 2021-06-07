import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import pandas as pd
import os
import footyviz

from moviepy import editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

RESOLUTION = 0.05

WIDTH, HEIGHT = 55 * 2, 36 * 2
FPS = 15

HX, HZ = 52.5, 34

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255),
              (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
              (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128),
              (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0),
              (128, 0, 128)]

team_color_list = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

plane = np.array([0, 1, 0])
origin = np.array([0, 0, 0])
theta = np.linspace(0, 2. * np.pi, 50)
alpha = np.linspace(0.72 * np.pi, 1.28 * np.pi, 25)
pitch = [
    # home half field
    np.array([
        [0, 34, 0, 1],
        [0, -34, 0, 1],
        [52.5, -34, 0, 1],
        [52.5, 34, 0, 1],
        [0, 34, 0, 1],
    ]).T,

    # home penalty area
    np.array([
        [52.5, -20.15, 0, 1],
        [52.5, 20.15, 0, 1],
        [36, 20.15, 0, 1],
        [36, -20.15, 0, 1],
        [52.5, -20.15, 0, 1],
    ]).T,

    # home goal area
    np.array([
        [52.5, -9.15, 0, 1],
        [52.5, 9.15, 0, 1],
        [47, 9.15, 0, 1],
        [47, -9.15, 0, 1],
        [52.5, -9.15, 0, 1],
    ]).T,

    # away half field
    np.array([
        [0, 34, 0, 1],
        [0, -34, 0, 1],
        [-52.5, -34, 0, 1],
        [-52.5, 34, 0, 1],
        [0, 34, 0, 1],
    ]).T,

    # away penalty area
    np.array([
        [-52.5, -20.15, 0, 1],
        [-52.5, 20.15, 0, 1],
        [-36, 20.15, 0, 1],
        [-36, -20.15, 0, 1],
        [-52.5, -20.15, 0, 1],
    ]).T,

    # away goal area
    np.array([
        [-52.5, -9.15, 0, 1],
        [-52.5, 9.15, 0, 1],
        [-47, 9.15, 0, 1],
        [-47, -9.15, 0, 1],
        [-52.5, -9.15, 0, 1],
    ]).T,

    # center circle
    np.stack([
        9.15 * np.cos(theta), 9.15 * np.sin(theta),
        np.zeros_like(theta),
        np.ones_like(theta)
    ]),

    # home circle
    np.stack([
        41.5 + 9.15 * np.cos(alpha), 9.15 * np.sin(alpha),
        np.zeros_like(alpha),
        np.ones_like(alpha)
    ]),

    # away circle
    np.stack([
        -41.5 - 9.15 * np.cos(alpha), 9.15 * np.sin(alpha),
        np.zeros_like(alpha),
        np.ones_like(alpha)
    ]),
]


def visualize_tracks_on_pitch(tracks, gt=None, invert=False):
    W, H = int(WIDTH // RESOLUTION), int(HEIGHT // RESOLUTION)
    cx, cz = WIDTH / 2, HEIGHT / 2

    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, :] = [7, 124, 52]
    # print(img)
    pitch_lines = [
        (np.vstack([pts[0] + cx, pts[1] + cz]) // RESOLUTION).T.reshape(
            (-1, 1, 2)).astype(np.int32) for pts in pitch
    ]
    cv2.polylines(img, pitch_lines, False, (255, 255, 255), 3)

    if gt is not None:
        start_frame, end_frame = np.min(gt[:, 0]), np.max(gt[:, 0])

    else:
        start_frame, end_frame = np.min(tracks[:, 0]), np.max(tracks[:, 0])

    frames = np.arange(start_frame, end_frame)

    # prev_frame = start_frame

    for frame in frames:
        tracks_cur = tracks[tracks[:, 0] == frame]

        if frame != start_frame:
            print('write frame %d' % frame)
            if out is not None:
                videoWriter.write(out)
            else:
                videoWriter.write(img)
            # initialize a new background for every frame
            img = np.zeros((H, W, 3), np.uint8)
            img[:, :, :] = [7, 124, 52]
            # plot pitch
            cv2.polylines(img, pitch_lines, False, (255, 255, 255), 3)

        for track in tracks_cur:
            _, track_id, x, z, teamid, objid = track
            if invert:
                x = -x

            # draw individual points with id
            cv2.putText(img,
                        str(int(track_id)) + '(' + str(int(objid)) + ')', (int(
                            (x + cx) // RESOLUTION), int(
                                (z + cz) // RESOLUTION) - 8),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        team_color_list[int(teamid)], 2)
            

            cv2.circle(img, (int(
                (x + cx) // RESOLUTION), int((z + cz) // RESOLUTION)),
                    radius=10,
                    color=team_color_list[int(teamid)],
                    thickness=-1)
                
        out = None
        # draw ground truth
        if gt is not None:
            # find ground truth corresponds to the same frame
            gt_frame = gt[gt[:, 0] == frame]
            if gt_frame.shape[0] > 0:
                overlay = np.zeros((H, W, 3), np.uint8)
                for pl in gt_frame:
                    
                    # frame_id, jersey_number, X, Z, team_id, player_id
                    cv2.circle(overlay, (int((pl[2] + cx) // RESOLUTION),
                                        int((pl[3] + cz) // RESOLUTION)),
                            radius=30,
                            color=team_color_list[int(pl[4])],
                            thickness=-1)
                # blend ground truth overlay with the pitch
                out = cv2.addWeighted(img, 1.0, overlay, 0.5, 1)
        # prev_frame = frame


def convert_to_footyviz(tracks):
    # frame_id, track_id, x, z, teamid, objid
    # start_frame, end_frame = np.min(tracks[:,0]), np.max(tracks[:,0])
    # frames = np.arange(start_frame, end_frame)
    # frames_missing = frames[~np.isin(frames, tracks[:,0])]
    df = pd.DataFrame(tracks)
    df.columns = ['frame', 'player', 'x', 'y', 'teamid', 'objid']
    # print(df.columns)
    df = df.set_index('frame')
    # print(len(df.index.tolist()))
    # df = df.reindex(df.index.tolist() + list(frames_missing))
    df.x = 100 * (df.x + HX) / (2 * HX)
    df.y = 100 * (df.y + HZ) / (2 * HZ)
    df['team'] = 'others'
    # print(df.columns)
    df.team[df.teamid == 0] = 'defense'
    df.team[df.teamid == 1] = 'attack'
    df.team[df.teamid == 2] = 'referee'
    df['bgcolor'] = 'black'
    df['edgecolor'] = 'black'
    df.bgcolor[df.teamid == 0] = 'yellow'
    df.edgecolor[df.teamid == 0] = 'brown'
    df.bgcolor[df.teamid == 1] = 'red'
    df.edgecolor[df.teamid == 1] = 'white'
    df.bgcolor[df.teamid == 2] = 'blue'
    df.edgecolor[df.teamid == 2] = 'white'
    df['player_num'] = np.NaN
    df['z'] = 0

    # print(df['team'])
    # print(df.head())
    # print(df['x'].max(axis=0), df['x'].min(axis=0))
    # print(df['y'].max(axis=0), df['y'].min(axis=0))
    return df


def draw_frame_x(df, t, fps=FPS, voronoi=False):
    print(t)
    fig, ax, dfFrame = footyviz.draw_frame(df, t=t, fps=fps)
    if voronoi and (dfFrame is not None):
        fig, ax, dfFrame = footyviz.add_voronoi_to_fig(fig, ax, dfFrame)
    image = mplfig_to_npimage(fig)
    plt.close()
    return image


def make_animation(df, fps=FPS, voronoi=False):
    #calculated variables
    length = (df.index.max() + 20) / fps
    # clip = mpy.VideoClip(lambda x: print(x), duration=length-1).set_fps(fps)
    clip = mpy.VideoClip(
        lambda x: draw_frame_x(df, t=x, fps=fps, voronoi=voronoi),
        duration=length - 1).set_fps(fps)
    return clip


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--result_file",
                   type=str,
                   help="path to the tracking result",
                   required=True)
    a.add_argument("--ground_truth",
                   type=str,
                   help="path to the ground truth tracks",
                   default=None)
    a.add_argument("--fixed_cam", action='store_true', help="Whehter is fixed camera or not")
    a.add_argument("--viz", action='store_true', help="Whether to visualize the sport analysis or not")
    opt = a.parse_args()

    output_file = os.path.splitext(opt.result_file)[0]+'.mp4'
    track_res = np.genfromtxt(opt.result_file, delimiter=',')
    print(track_res.shape)
    frames = np.unique(track_res[:, 0])
    # print(frames.shape)
    # print(np.where(frames[1:]-frames[:-1] > 1))
    # print(np.max(track_res[:,0]), np.min(track_res[:,0]))
    # input("...")

    id0 = track_res[:, -2] == 0
    id1 = track_res[:, -2] == 1
    track_res[id0, -2] = 1
    track_res[id1, -2] = 0

    gt = None
    if opt.ground_truth:
        gt = np.genfromtxt(opt.ground_truth, delimiter=',')
        output_file = os.path.splitext(output_file)[0]+'_gt.mp4'
        # print(gt.shape)

    videoWriter = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS,
        (int(WIDTH // RESOLUTION), int(HEIGHT // RESOLUTION)))

    # visualize tracking results together with ground truth
    visualize_tracks_on_pitch(track_res, gt, opt.fixed_cam)

    if opt.viz:
        # visualize basic result with footyviz
        print("visualize sport analysis with voronoi")
        if opt.fixed_cam:
            # invert x direction
            track_res[:,2] = -track_res[:,2]
        df = convert_to_footyviz(track_res)
        # print(df.loc[0])
        # input("...")
        # fig, ax, dfFrame = footyviz.draw_frame(df, t=0.1, fps=25)
        # fig, ax, dfFrame = footyviz.add_voronoi_to_fig(fig, ax, dfFrame)
        # plt.show()
        clip = make_animation(df, voronoi=True)
        clip.ipython_display()
        clip.write_videofile(os.path.splitext(output_file)[0]+'_voronoi.mp4')
