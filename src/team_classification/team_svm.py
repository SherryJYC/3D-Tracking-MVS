
#frame: video frame; bbox: x1, y1, x2, y2
#team: 0 white; 1 red; 2 referee; 3 nothing; 4 goalkeeper

import numpy as np
import joblib

svm_path = "./team_svm.model"

def team_classification(frame, bbox)

    img = frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    RGB_feature = [np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])]
    model = joblib.load(svm_path)
    team = model.predict(RGB_feature)
    
return team

