from mmcv import Config
import os
from mmdet.apis import init_detector, inference_detector, show_result
import datetime

config_file="configs/cascade_rcnn_x101_64x4d_fpn_1x.py"

checkpoint_file ="work_dirs/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth" # normal coco

print("initialize model.")
model = init_detector(config_file, checkpoint_file, device='cuda:0')


