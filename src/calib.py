import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
import cv2
import numpy as np
import os
import math
import argparse
import json

SCALE = 1
RESOLUTION = 0.1
WIDTH, HEIGHT = 55 * 2, 36 * 2
FPS = 15
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255),
              (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
              (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128),
              (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0),
              (128, 0, 128)]
team_color_list = [(255,255,255), (0,0,255), (0,255,0), (255,0,0)]

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
        pts[[1, 2]] = pts[[2, 1]]
        x, y = project(pts, P)
        plt.plot(x, y, 'r-')

    plt.show()


def Rx(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    A = np.array([[1, 0, 0], [0, rcos, -rsin], [0, rsin, rcos]])
    return A


def Ry(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    #A = np.array([[rcos, 0, rsin],
    #              [0, 1, 0],
    #              [-rsin, 0, rcos]])
    K = np.array([[rcos, 0, -rsin], [0, 1, 0], [rsin, 0, rcos]])
    return K


def computeP(line, cx, cy):
    P = np.empty([3, 4])

    theta, phi, f, Cx, Cy, Cz = line
    R = Rx(phi).dot(
        Ry(theta).dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])))
    T = -R.dot(np.array([[Cx], [Cy], [Cz]]))
    K = np.eye(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, cx, cy
    P = np.dot(K, np.hstack((R, T)))

    return P


def cross(e):
    return np.array([[0, e[2], -e[1]], [-e[2], 0, e[0]], [e[1], -e[0], 0]])


def skew(w):
    top = np.hstack([-w[3] * np.diag([1, 1, 1]), cross(w[:3])])
    bottom = np.hstack([w[:3], np.zeros(3)])

    return np.vstack([top, bottom])


def point_normal_eq(normal, pt):
    return np.hstack([normal, -np.inner(normal, pt)]).reshape(-1, 1)


def backproject_pitch(P, x, C_cam):
    X = np.dot(np.linalg.pinv(P), x)
    X /= X[-1]

    C_homo = np.hstack([C_cam, 1]).reshape(-1, 1)
    p = np.dot(X.T, plane_normal) * C_homo - np.dot(C_homo.T, plane_normal) * X

    p /= p[-1]
    return p.ravel()


def display_soccer_pitch_ground(points, imgname, C_cam):
    plt.figure(figsize=(16, 9))

    for pts in pitch:
        pts[[1, 2]] = pts[[2, 1]]
        px, py = pts[0], pts[2]
        plt.plot(SCALE * px, SCALE * py, 'r-')

    plt.scatter(C_cam[0], C_cam[1])
    plt.annotate('Camera', (C_cam[0], C_cam[1]))

    for p in points:
        plt.scatter(p[0], p[1])
        plt.annotate(p[2], (p[0], p[1]))

    plt.savefig(imgname)
    plt.show()

def display_cameras_on_pitch(bboxs, cams, keys, imgname):
    fig, ax = plt.subplots(1,1,figsize=(16, 9))

    for pts in pitch:
        pts[[1, 2]] = pts[[2, 1]]
        px, py = pts[0], pts[2]
        plt.plot(SCALE * px, SCALE * py, 'r-')
    
    patches = []
    # colors = iter([plt.cm.Paired(i) for i in range(len(cams))])
    colors = []
    for bbox, cam, key in zip(bboxs, cams, keys):
        # clr = next(colors)
        
        if int(key) % 4 == 0:
            clr = 'red'
        elif int(key) % 4 == 1:
            clr = 'gray'
        elif int(key) % 4 == 2:
            clr = 'red'
        else:
            clr = 'gray'
        plt.scatter(cam[0], cam[1], c=clr)
        plt.annotate('Camera '+key, (cam[0], cam[1]))

        # plot the lines
        for coord in bbox:
            # line = np.vstack([cam, coord])
            plt.plot([cam[0], coord[0]], [cam[1], coord[1]], color=clr, linestyle=':', alpha=0.8)

        # plot field of view
        polygon = Polygon(bbox, True)
        patches.append(polygon)
        colors.append(clr)
    
    # p = PatchCollection(patches, alpha=0.4, cmap=plt.cm.Paired)
    # p.set_array(np.arange(len(cams)))
    p = PatchCollection(patches, alpha=0.1, color='gray')
    # p.set_array(np.arange(len(cams)))
    ax.add_collection(p)
    p.set_color(colors)
    p.set_alpha(list(map(lambda x: 0.2 if x == 'gray' else 0.1, colors)))
    ax.set_xbound(-85, 85)
    ax.set_ybound(-60,60)

    plt.savefig(imgname)
    plt.show()


def visualize_tracks_on_pitch(tracks):
    W, H = int(WIDTH // RESOLUTION), int(HEIGHT // RESOLUTION)
    cx, cz = WIDTH / 2, HEIGHT / 2

    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, :] = [7, 124, 52]
    print(img)
    pitch_lines = [
        (np.vstack([pts[0] + cx, pts[1] + cz]) // RESOLUTION).T.reshape(
            (-1, 1, 2)).astype(np.int32) for pts in pitch
    ]
    cv2.polylines(img, pitch_lines, False, (255, 255, 255), 3)

    prev_frame = tracks[0, 0]

    for track in tracks:
        frame_id, track_id, x, z, teamid, objid = track
        # print(frame_id)
        if frame_id != prev_frame:
            print('write frame %d' % frame_id)
            videoWriter.write(img)
            img = np.zeros((H, W, 3), np.uint8)
            img[:, :, :] = [7, 124, 52]
            # plot pitch
            cv2.polylines(img, pitch_lines, False, (255, 255, 255), 3)
        # draw individual points with id
        cv2.putText(img, str(int(track_id))+'('+str(int(objid))+')', (int(
            (x + cx) // RESOLUTION), int(
                (z + cz) // RESOLUTION) - 8), cv2.FONT_HERSHEY_PLAIN, 1,
                    team_color_list[int(teamid)], 2)
        cv2.circle(img, (int(
            (x + cx) // RESOLUTION), int((z + cz) // RESOLUTION)),
                   radius=5,
                   color=team_color_list[int(teamid)],
                   thickness=-1)
        prev_frame = frame_id


if __name__ == "__main__":

    a = argparse.ArgumentParser()
    a.add_argument('--calib_path', type=str, help='Path to the calibration file')
    a.add_argument('--res_path', type=str, help='Path to the tracking result')
    a.add_argument('--xymode', action='store_true', help='Whether to use XY mode/WH mode to parse the tracking result')
    a.add_argument('--reid', action='store_true', help='Include reid in the tracking result')
    a.add_argument('--viz', action='store_true', help='Whether to visualize the tracking result')

    opt = a.parse_args()

    img = cv2.imread(
        "/scratch2/wuti/Others/3DVision/fixed_cameras/FIXED-0125-0135/cam1_img/image0001.png"
    )

    H, W, _ = img.shape
    print(W, H)

    corners = [[0, 0, 1], [0, H, 1], [W, H, 1], [W, 0, 1]]
    # cam_list = ["1","2","3","4","5","6","7","8"]
    cam_list = ["2","5","6"]

    # result_file = '/scratch2/wuti/Others/3DVision/test_result_filtered_team/16m_right_filtered_team.txt'
    result_file = opt.res_path

    if opt.reid:
        tracks = np.genfromtxt(result_file,
                            delimiter=',',
                            usecols=(0, 1, 2, 3, 4, 5, 10)).astype(int)
        outname = result_file.replace('.txt','_pitch_reid.txt')

    else:
        tracks = np.genfromtxt(result_file,
                            delimiter=',',
                            usecols=(0, 1, 2, 3, 4, 5)).astype(int)
        outname = result_file.replace('.txt','_pitch.txt')


    # calib_file = '/scratch2/wuti/Others/3DVision/calibration_results/0125-0135/RIGHT/calib.txt'
    calib_file = opt.calib_path
    if calib_file.endswith('.json'):
        with open(calib_file) as f:
            calib = json.load(f)
        # get corresponding camera calibrations
        imgname = os.path.basename(result_file).split('.')[0].split('_')[-1]
        K = np.array(calib[imgname]["K"]).reshape(3,3)
        R = np.array(calib[imgname]["R"]).reshape(3,3)
        T = np.array(calib[imgname]["T"]).reshape(3,1)
        P = K @ np.hstack([R, T])
        C_cam = -R.T.dot(T).ravel()
        print(C_cam)
        plane = np.array([0, 0, 1])
        # input("....")
    else:
        calib = np.genfromtxt(calib_file,
                            delimiter=',',
                            usecols=(1, 2, 3, 4, 5, 6))
        imgname = np.genfromtxt(calib_file, delimiter=',', usecols=(7), dtype=str)
        framecalib = [int(x.split('.')[0][5:]) for x in imgname]
        # print(imgname)
        cx, cy = 960, 540
        Projections = [computeP(calibline, cx, cy) for calibline in calib]
        print(len(Projections), len(calib))
        plane = np.array([0, 1, 0])

    plane_normal = point_normal_eq(plane, origin)
    
    # save tracking results coordinates on the pitch
    tracks_pitch = []
    # points = []
    # tracks = tracks[tracks[:,0] == 1]
    # print(Projections)
    for track in tracks:
        if opt.xymode:
            x1, y1, x2, y2 = track[2], track[3], track[4], track[5]
        else:
            x1, y1, x2, y2 = track[2], track[3], track[2]+track[4], track[3]+track[5]
        # get the calibration of the corresponding frames
        # print(framecalib == track[0])
        if not calib_file.endswith('.json'):
            if track[0] not in framecalib:
                continue
            P = Projections[framecalib.index(track[0])]
            C_cam = calib[framecalib.index(track[0]),-3:]

        tx, ty, tz, _ = backproject_pitch(P,np.array([(x1+x2)/2,y2,1]).reshape(-1,1),C_cam)

        if not calib_file.endswith('.json'):
            ty = tz
        
        # points.append([tx, ty, track[1]])

        # frame id, track id, x, z, team id
        if opt.reid:
            tracks_pitch.append([track[0], track[1], tx, ty, track[-1]])
        else:
            tracks_pitch.append([track[0], track[1], tx, ty])
    
    # print(points)
    # display_soccer_pitch_ground(points, 'test.png', C_cam)

    np.savetxt(outname, np.array(tracks_pitch), delimiter=',')

    if calib_file.endswith('.json'):
        # visualize cameras
        if opt.viz:
            cams = []
            bboxs = []
            keys = []
            for key, calib_cam in calib.items():
                if key not in cam_list:
                    continue
                # print(key)
                keys.append(key)
                K = np.array(calib_cam["K"]).reshape(3,3)
                R = np.array(calib_cam["R"]).reshape(3,3)
                T = np.array(calib_cam["T"]).reshape(3,1)
                P = K @ np.hstack([R, T])
                cam = -R.T.dot(T).ravel()
                cams.append(cam)
                bbox = []
                for corner in corners:
                    tx, ty, tz, _ = backproject_pitch(P,np.array(corner).reshape(-1,1),cam)
                    bbox.append([tx, ty])
                bboxs.append(np.array(bbox))

            display_cameras_on_pitch(bboxs, cams, keys, 'fixed_cameras_on_pitch_256.png')

    # # visualize tracking result
    # # result_file = '/scratch2/wuti/Others/3DVision/cam1_result_filtered_team/cam1_right_team.txt'
    # result_file = '/scratch2/wuti/Others/3DVision/ground_truth/ground_truth_3d/gt_pitch.txt'

    # output_file = result_file.replace('.txt', '.mp4')
    # videoWriter = cv2.VideoWriter(
    #     output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS,
    #     (int(WIDTH // RESOLUTION), int(HEIGHT // RESOLUTION)))
    # track_res = np.genfromtxt(result_file, delimiter=',')
    # print(track_res.shape)
    # visualize_tracks_on_pitch(np.array(tracks_pitch))