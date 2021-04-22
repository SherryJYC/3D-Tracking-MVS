import numpy as np
import cv2
from ainvr import ainvr
from iou_tracker.iou_tracker import Tracker

font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8





def render_detections(im, detections):
    for detection in detections:
    	xmin = int(detection["xmin"])
    	ymin = int(detection["ymin"])
    	xmax = int(detection["xmax"])
    	ymax = int(detection["ymax"])
    	label = detection["class"]
        width = xmax - xmin
        height = ymax - ymin
	
 
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im



def render_tracks(im, tracks):
    for id_, track in tracks.items():
        xmin, ymin, xmax, ymax = track['bbox']
        xmin = int(xmin)
        ymin = int(ymin)
        width = int(xmax - xmin)
        height = int(ymax - ymin)
        label = str(id_)
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im

  


def parse_detections(detections):
    parsed_detections = []
    for detection in detections:
	score = detection["score"]
	xmin = detection['xmin']
	xmax = detection['xmax']
	ymin = detection['ymin']
	ymax = detection['ymax'] 
	bbox = (xmin, ymin, xmax,  ymax)
	parsed_detections.append({"bbox":bbox, "score":score})
	
    return parsed_detections








cap = cv2.VideoCapture("rtsp://221.120.30.101/live.sdp")
tracker = Tracker(t_max = 30)





while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()
    if frame is None:
	print("Empty frame received!")
	continue
    detections = ainvr.detect_img(frame)
    parsed_detections = parse_detections(detections)
    tracks = tracker.track(parsed_detections)
    print(tracks)
    
    frame = render_tracks(frame, tracks)
    #for i in range(5):
    #	cap.grab()
   

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
