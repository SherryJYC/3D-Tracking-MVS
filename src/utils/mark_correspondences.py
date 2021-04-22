import argparse
import os
import numpy as np
import cv2

img1_coords = []
img2_coords = []
def get_coordinates(event, x, y, flags, param):
    print("event: ",event)
    if event == cv2.EVENT_LBUTTONDOWN:
    # if event == cv2.EVENT_LBUTTONDBLCLK:
        # print(param)
        if param == "img1":
            print("img1 mouse clicked")

            img1_coords.append([x,y])
            cv2.drawMarker(img1,(x,y),(0,255,0),thickness=2)
            cv2.imshow("image 1",img1)
        if param == "img2":
            print("img2 mouse clicked")

            img2_coords.append([x,y])
            cv2.drawMarker(img2,(x,y),(0,0,255),thickness=2)
            cv2.imshow("image 2", img2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, help='Path to the first image')
    parser.add_argument('--img2', type=str, help='Path to the second image')
    parser.add_argument('--out_dir', type=str, help='Path to store the correspondences')

    opt = parser.parse_args()
    print(opt)

    img1 = opt.img1    
    img2 = opt.img2

    out_name = os.path.basename(img1).split('.')[0]+'_'+os.path.basename(img2).split('.')[0]+'_img_coords.npy'
    out_name_homo = os.path.basename(img1).split('.')[0]+'_'+os.path.basename(img2).split('.')[0]+'_homo.npy'

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    print(img1.shape)
    print(img2.shape)
    # change color space
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

    cv2.namedWindow("image 1")
    cv2.setMouseCallback("image 1",get_coordinates, "img1")
    cv2.namedWindow("image 2")
    cv2.setMouseCallback('image 2',get_coordinates, "img2")

    print("set mouse callback")

    # open the first image
    while True:
        cv2.imshow("image 1", img1)
        # open the second image
        cv2.imshow("image 2", img2)

        # setup mouse click callback
        key = cv2.waitKey(1) & 0xFF
        
        # break the loop when the `q` key was pressed
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    # save points to directory
    img1_coords = np.array(img1_coords)
    img2_coords = np.array(img2_coords)

    print(img1_coords)
    print(img2_coords)

    coords = np.vstack([img1_coords,img2_coords])

    assert img1_coords.shape == img2_coords.shape
    homo, mask = cv2.findHomography(img1_coords, img2_coords)
    print(homo)

    with open(os.path.join(opt.out_dir, out_name), 'wb') as f:
        np.save(f, coords)

    with open(os.path.join(opt.out_dir, out_name_homo), 'wb') as f:
        np.save(f, homo)
