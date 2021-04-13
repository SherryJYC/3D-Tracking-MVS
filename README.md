# 3D-Tracking-MVS
Course project for 3DV 2021 Spring @ ETH Zurich

## Preprocessing
- Split video into image frames
```
python src/utils/v2img.py --pathIn=data/0125-0135/CAM1/CAM1.mp4 --pathOut=data/0125-0135/CAM1/img --splitnum=1
```
- Handle moving cameras
```
python src/utils/mov2static.py --calib_file=data/calibration_results/0125-0135/CAM1/calib.txt --img_dir=data/0125-0135/CAM1/img --output_dir=data/0125-0135/CAM1/img_static
```

- After processing, data folder structure:
```
data
├── 0125-0135
│   ├── CAM1
│   │   ├── img
│   │   └── img_static
│   └── RIGHT
│       └── img
└── calibration_results
    └── 0125-0135
        ├── CAM1
        └── RIGHT
```

## Generate tracklet

## Link short tracklets

## Cross-camera link


## Useful literature

- Learning to Track and Identify Players from Broadcast Sports Videos 2012 ::rainbow:: [[paper](https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf)]
- Multicamera people tracking with probabilistic occupancy map 2013 [[paper](https://infoscience.epfl.ch/record/145991)][[project](https://www.epfl.ch/labs/cvlab/research/research-surv/research-body-surv-index-php/)]
- Multi-camera multi-player tracking with deep player identification in sports video 2020 [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320300650)]

## Useful Github repo
[Full pipeline: POM+DeepOcculusion+pyKSP](https://www.epfl.ch/labs/cvlab/research/research-surv/research-body-surv-index-php/) <br/>
[Soccer player tracking system](https://github.com/AndresGalaviz/Football-Player-Tracking) <br/>
[CVPR2020: How To Train Your Deep Multi-Object Tracker](https://github.com/yihongXU/deepMOT)
