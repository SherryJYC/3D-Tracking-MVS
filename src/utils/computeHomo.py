import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# COURT_SIZE = ()
img_coords = []
world_coords = []
def get_coordinates(event, x, y, flags, param):
    # print("event: ",event)
    if event == cv2.EVENT_LBUTTONDOWN:
    # if event == cv2.EVENT_LBUTTONDBLCLK:
        # print(param)
        print("img mouse clicked")
        img_coords.append([x,y])
        cv2.drawMarker(img,(x,y),(0,255,0),thickness=2)
        cv2.imshow("image",img)

        # input world coordniates
        world = [float(coord) for coord in input("please enter the 3D coordinates of the %d point:\n" %len(world_coords)).split()]
        print(world)
        world_coords.append(world)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to the image')
    parser.add_argument('--out_dir', type=str, help='Path to store the correspondences and homography')

    opt = parser.parse_args()
    print(opt)

    img = opt.img   

    out_name = 'img_coords.npy'
    out_name_world = 'world_coords.npy'
    out_name_homo = 'homo.npy'

    img = cv2.imread(img)
    print(img.shape)

    # change color space
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image",get_coordinates)
    
    print("set mouse callback")

    while True:
        # open image
        cv2.imshow("image", img)

        # setup mouse click callback
        key = cv2.waitKey(1) & 0xFF
        
        # break the loop when the `q` key was pressed
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    # save points to directory
    img_coords = np.array(img_coords)
    world_coords = np.array(world_coords)

    print(img_coords)
    print(world_coords)

    assert img_coords.shape == world_coords.shape
    homo, mask = cv2.findHomography(img_coords, world_coords)
    print(homo)

    # des = cv2.warpPerspective(img, homo, (1500,1000))
    # plt.imshow(des)
    # plt.show()
    with open(os.path.join(opt.out_dir, out_name), 'wb') as f:
        np.save(f, img_coords)
    
    with open(os.path.join(opt.out_dir, out_name_world), 'wb') as f:
        np.save(f, world_coords)

    with open(os.path.join(opt.out_dir, out_name_homo), 'wb') as f:
        np.save(f, homo)
