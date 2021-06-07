
#frame: video frame; bbox: x1, y1, x2, y2
#team: 0 white; 1 red; 2 referee; 3 nothing; 4 goalkeeper

import numpy as np
import joblib
import sys
import cv2

input_path = sys.argv[1]
image_path = sys.argv[2]
svm_path = "./team_svm.model"



def team_classification(frame, bbox)

    img = frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    RGB_feature = [np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])]
    model = joblib.load(svm_path)
    team = model.predict(RGB_feature)
    
return team


images = []
if "EPTS" in input_path:
    all_file = 551
else:
    all_file = 251
for i in range(all_file):
    image = cv2.imread(image_path+"/image"+str(i+1).zfill(4)+".png")
    if "EPTS" in input_path:
        image = image.reshape([2304,4096,3])
    else:
        image = image.reshape([1080,1920,3])
    images.append(image)

    
output_path = (path.split("/")[-1]).split(".")[0] + "_team.txt" 
of = open(output_path, "w")
f = open(input_path, "r")
lines = f.readlines()
for line in lines:
    line=line.strip('\n')
    ls = line.split(",")
    frame = int(ls[0])-1
    image = images[frame]
    x1 = float(ls[2])
    y1 = float(ls[3])
    x2 = float(ls[2]) + float(ls[4])
    y2 = float(ls[3]) + float(ls[5])
    bbox = np.array([x1, y1, x2, y2])
    bbox = bbox.astype(np.int)
    bbox[:2] = np.maximum(0, bbox[:2])    
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        print(line, bbox)
        team = ['3']
    else:
        team = team_classification(image, bbox)
        #print(team)
    newline = line+","+team[0]+'\n'
    of.write(newline)

