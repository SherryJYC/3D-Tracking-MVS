import cv2
import numpy as np


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        if digitize1:
            x1.append(x)
            y1.append(y)
            cv2.circle(img1, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image 1", img1)
            print(x, y, '1')
        else:
            x2.append(x)
            y2.append(y)
            cv2.circle(img1, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img1, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image 2", img2)
            print(x, y, '2')

def main(img1, img2, digitize1):
    # digitize image 1
    digitize1 = True
    cv2.namedWindow("image 1")
    cv2.setMouseCallback("image 1", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image 1", img1)
    cv2.waitKey(0)

    # digitize image 2
    digitize1 = False
    cv2.namedWindow("image 2")
    cv2.setMouseCallback("image 2", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image 2", img2)
    cv2.waitKey(0)

if __name__ == "__main__":
    # Picture path
    img1path = './data/trial.png'
    img2path = './data/trial.png'
    img1 = cv2.imread(img1path)
    img2 = cv2.imread(img2path)

    digitize1 = True
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    main(img1, img2, digitize1)