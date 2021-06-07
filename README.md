# 3D Player Tracking with Multi-View Stream
Project for 3DV 2021 Spring @ ETH Zurich [[Report Link]()] <br/>
<br/>
This repo contains a full pipeline to support 3D position tracking of soccer players, with multi-view calibrated moving/fixed video sequences as inputs. 
- In single-camera tracking stage, Tracktor++ is used to get 2D positions.
- In multi-camera tracking stage, 2D positions are projected into 3D positions. Then across-camera association is achieved as an optimization problem with spatial, temporal and visual constraints.
- In the end, visualization in 2D, 3D and a voronoi visualization for sports coaching purpose are provided.

3D Tracking            |  Sports Coaching 
:-------------------------:|:-------------------------:
<img alt="demo" src="https://github.com/SherryJYC/3D-Tracking-MVS/blob/main/misc/cam1_right_team.gif" width="400" height="250" />  |  <img alt="demo" src="https://github.com/SherryJYC/3D-Tracking-MVS/blob/main/misc/cam1_right_team_gt_voronoi.gif" width="400" height="250" />

## Demo
To run the demos, please download the required data from [this link](https://polybox.ethz.ch/index.php/s/SrBn2DtKEJQaWFg) and place the folder inside this repository
- Run the demo visualization on the moving cameras
```shell
bash script/demo_moving.sh
```
- Run the demo visualization on the fixed cameras
```shell
bash script/demo_fix.sh
```

## Preprocessing 
- Split video into image frames
```
python src/utils/v2img.py --pathIn=data/0125-0135/CAM1/CAM1.mp4 --pathOut=data/0125-0135/CAM1/img --splitnum=1
```
- Estimate football pitch homography (size 120m * 90m [ref](https://www.quora.com/What-are-the-official-dimensions-of-a-soccer-field-in-the-FIFA-World-Cup))
> [FIFA official document](https://img.fifa.com/image/upload/datdz0pms85gbnqy4j3k.pdf)

```
python src/utils/computeHomo.py --img=data/0125-0135/RIGHT/img/image0000.jpg --out_dir=data/0125-0135/RIGHT/
```

- Handle moving cameras
```
python src/utils/mov2static.py --calib_file=data/calibration_results/0125-0135/CAM1/calib.txt --img_dir=data/0125-0135/CAM1/img --output_dir=data/0125-0135/CAM1/img_static
```
- Convert ground truth/annotation json to text file
```
python src/utils/json2txt.py --jsonfile=data/0125-0135/0125-0135.json
```

<!--
- After processing, data folder structure should be like:
```
data
├── 0125-0135
│   ├── CAM1
│   │   ├── img
│   │   ├── img_static
│   │   └── homo.npy
│   ├── RIGHT
│   │   
│   ├── proj_config.txt
│   ├── 16m_left.txt
│   ├── 16m_right.txt
│   └── id_mapping.txt
│       
└── calibration_results
    └── 0125-0135
        ├── CAM1
        └── RIGHT
```
-->
- [Download preprocessed data](https://polybox.ethz.ch/index.php/s/CvcT5pxOY90bpIF)
> only include homography and config files, large image folder not included

## Single-camera tracking
- Train Faster RCNN
```
Jiao Ying ???
```
- Run Tracktor
```
Jiao Ying ???
```
- Run ReID(team id) model
```
Jiao Ying ???
```
- Convert tracking results to coordinates on the pitch
> Equation to find the intersection of a line with a plane ([ref](https://math.stackexchange.com/questions/2041296/algorithm-for-line-in-plane-intersection-in-3d))

```shell
python src/calib.py --calib_path=PATH_TO_CALIB --res_path=PATH_TO_TRACKING_RESULT --xymode --reid

# also plot the camera positions for fixed cameras
python src/calib.py --calib_path=PATH_TO_CALIB --res_path=PATH_TO_TRACKING_RESULT --viz
```
## Across-camera association

- Run two-cam tracker
```
python src/runMCTRacker.py 

# add team id constraint
python src/runMCTRacker.py --doreid
```

- Run multi-cam tracker (e.g. 8 cams)
```
python src/runTreeMCTracker.py --doreid
```

## Evaluation

- Produce quatitative results (visualize results)
> visualize 2d bounding box

```shell
# if format <x, y, w, h>
python src/utils/visualize.py --img_dir=data/0125-0135/RIGHT/img --result_file=output/tracktor/16m_right_prediction.txt 
# if format <x1, y1, x2, y2>
python src/utils/visualize.py --img_dir=data/0125-0135/RIGHT/img --result_file=output/iou/16m_right.txt --xymode
# if with team id
python src/utils/visualize.py --img_dir=data/0125-0135/RIGHT/img --result_file=output/tracktor/16m_right_prediction.txt --reid
# if 3d mode
python src/utils/visualize.py --img_dir=data/0125-0135/RIGHT/img --result_file=output/tracktor/RIGHT.txt --calib_file=data/calibration_results/0125-0135/RIGHT/calib.txt  --pitchmode
```
> visualize 3d tracking result with ground truth and voronoi diagram

```
python src/utils/visualize_on_pitch.py --result_file=PATH_TO_TRACKING_RESULT --ground_truth=PATH_TO_GROUND_TRUTH
```
> visualize 3d ground truth on camera frames (reprojection)

```
python src/utils/visualize_tracab --img_path=PATH_TO_IMAGES --calib_path=PATH_TO_CALIB --gt_path=PATH_TO_TRACAB_GT --output_path=PATH_TO_OUTPUT_VIDEO
```
- Produce quantitative result

```
# 2d <frame id, objid, x, y, w, h, .., ...>
python src/motmetrics/apps/eval_motchallenge.py data/0125-0135/ output/tracktor_filtered

# 3d
python src/utils/eval3d.py --pred=output/pitch/EPTS_3_pitch.txt_EPTS_4_pitch.txt.txt --fixcam  --gt=data/fixedcam/gt_pitch_550.txt
python src/utils/eval3d.py --fixcam --boxplot
```

<!--
## Result

> 0125-0135 right moving camera
```
SORT
         IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM   MOTA  MOTP IDt IDa IDm
RIGHT   58.3% 49.4% 71.3% 83.4% 57.8% 25 15 10  0 3674 1001  27  181  22.0% 0.340   5  23   1
CAM1    43.8% 35.3% 57.5% 67.9% 41.7% 22  6 15  1 2786  942  36  112 -28.2% 0.360   7  27   0
OVERALL 53.3% 44.4% 66.8% 78.3% 52.1% 47 21 25  1 6460 1943  63  293   5.5% 0.346  12  50   1

IOU
         IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM   MOTA  MOTP IDt IDa IDm
RIGHT   61.8% 53.3% 73.6% 82.9% 60.0% 25 17  8  0 3330 1029  46  279  26.9% 0.336  15  23   1
CAM1    42.6% 33.2% 59.2% 67.7% 38.0% 22  6 15  1 3238  948  53  144 -44.4% 0.355   7  31   0
OVERALL 54.9% 45.6% 68.9% 77.9% 51.5% 47 23 23  1 6568 1977  99  423   3.6% 0.341  22  54   1

Tracktor
         IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM   MOTA  MOTP IDt IDa IDm
RIGHT   72.6% 69.7% 75.8% 85.9% 79.0% 25 19  6  0 1376  848  21  213  62.8% 0.322   7  15   1
CAM1    49.3% 39.4% 65.9% 69.4% 41.5% 22  6 15  1 2871  899  17  146 -29.0% 0.344  10   7   0
OVERALL 63.7% 56.7% 72.6% 80.5% 63.0% 47 25 21  1 4247 1747  38  359  32.7% 0.328  17  22   1

```
-->

## Acknowledgement
We would like to thank the following Github repos or softwares: <br/>
- [Supervisely](https://app.supervise.ly/init)
- [Tracktor++](https://github.com/phil-bergmann/tracking_wo_bnw)
- [IoU tracker](https://github.com/GBJim/iou_tracker)
- [SORT](https://github.com/abewley/sort)
- [Football crunching (footyviz)](https://medium.com/football-crunching)
- [Pymotmetrics](https://github.com/cheind/py-motmetrics)

## Authors
Yuchang Jiang, Tianyu Wu, Ying Jiao, Yelan Tao

<!--
## Useful literature

- Learning to Track and Identify Players from Broadcast Sports Videos 2012 :rainbow:[[paper](https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf)]
- Multicamera people tracking with probabilistic occupancy map 2013 [[paper](https://infoscience.epfl.ch/record/145991)][[project](https://www.epfl.ch/labs/cvlab/research/research-surv/research-body-surv-index-php/)]
- Multi-camera multi-player tracking with deep player identification in sports video 2020 [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320300650)]

## Useful Github repo
[Full pipeline: POM+DeepOcculusion+pyKSP](https://www.epfl.ch/labs/cvlab/research/research-surv/research-body-surv-index-php/) <br/>
[Soccer player tracking system](https://github.com/AndresGalaviz/Football-Player-Tracking) <br/>
[CVPR2020: How To Train Your Deep Multi-Object Tracker](https://github.com/yihongXU/deepMOT)
-->
