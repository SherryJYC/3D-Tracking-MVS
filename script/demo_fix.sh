#!/bin/bash

# project 2d tracking result onto the pitch
printf "Project single camera tracking result onto the pitch\n"
for i in {1..8}
do
    printf "Projecting fixed camera $i...\n"
    python src/calib.py --calib_path data/calibrations/fixcam/calib.json --res_path data/tracks/fixcam/EPTS_${i}_team.txt --reid
done
printf "done.\n"
read -p "Press enter to continue"

# multi-camera tracking
printf "\n########################################\n"
printf "Run multi-camera tracking result"
python src/runTreeMCTracker.py --doreid
read -p "Press enter to continue"

# visualize multi-camera tracking
printf "\n########################################\n"
printf "Visualize multi-camera tracking result"
python src/utils/visualize_on_pitch.py --result_file /scratch2/wuti/Others/3DVision/submission/3D-Tracking-MVS/data/tracks/fixcam/results/results/results/EPTS_1_team_pitch_reid.txt_EPTS_2_team_pitch_reid.txt.txt_EPTS_3_team_pitch_reid.txt_EPTS_4_team_pitch_reid.txt.txt.txt_EPTS_5_team_pitch_reid.txt_EPTS_6_team_pitch_reid.txt.txt_EPTS_7_team_pitch_reid.txt_EPTS_8_team_pitch_reid.txt.txt.txt.txt --fixed_cam --viz