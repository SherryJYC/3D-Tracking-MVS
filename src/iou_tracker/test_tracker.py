from iou_tracker import Tracker
from util import load_mot, save_to_csv

DETS_PATH = "./MOT17/train/MOT17-04-SDP/det/det.txt"
detections = load_mot(DETS_PATH)
tracker = Tracker()
tracks = {}
for frame, detection in enumerate(detections, start=1):
	track = tracker.track(detection)
	tracks[frame] = track