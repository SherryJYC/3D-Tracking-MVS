#!/bin/bash

# extract camera images
printf "Extract frames from the video\n"
python src/utils/v2img.py --pathIn data/videos/movingcam/0125_0135/cam1.mp4 --pathOut data/videos/movingcam/0125_0135/imgs/cam1 --splitnum=1
read -p "Press enter to continue"

# visualize single camera tracking result
printf "\n########################################\n"
printf "Visualize single camera tracking result\n"
python src/utils/visualize.py --img_dir data/videos/movingcam/0125_0135/imgs/cam1 --result_file data/tracks/movingcam/cam1_filtered_team.txt --reid
read -p "Press enter to continue"

# project 2d tracking result onto the pitch
printf "\n########################################\n"
printf "Project single camera tracking result onto the pitch\n"
printf "Projecting cam1...\n"
python src/calib.py --calib_path data/calibrations/movingcam/0125_0135/cam1/calib.txt --res_path data/tracks/movingcam/cam1_filtered_team.txt --reid
printf "done.\n"
read -p "Press enter to continue"

printf "Projecting camera right...\n"
python src/calib.py --calib_path data/calibrations/movingcam/0125_0135/right/calib.txt --res_path data/tracks/movingcam/right_filtered_team.txt --reid
printf "done.\n"
read -p "Press enter to continue"

# multi-camera tracking
printf "\n########################################\n"
printf "Run multi-camera tracking result\n"
python src/runMCTracker.py --doreid
read -p "Press enter to continue"

# visualize multi-camera tracking
printf "\n########################################\n"
printf "Visualize multi-camera tracking result\n"
python src/utils/visualize_on_pitch.py --result_file /scratch2/wuti/Others/3DVision/submission/3D-Tracking-MVS/data/tracks/movingcam/cam1_right_team.txt --viz