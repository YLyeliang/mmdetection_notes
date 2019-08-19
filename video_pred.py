from mmcv import Config
import os
from mmdet.apis import init_detector, inference_detector, show_result
import cv2
import datetime

config_file="configs/cascade_rcnn_x101_64x4d_fpn_1x.py"

checkpoint_file ="work_dirs/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth" # normal coco

print("initialize model.")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

class video_io(object):
    """
    A video readline and writeline process,which can be modified by
    any other functions.
    """

    def __init__(self, read_path, write=None, model=None, video_type='mp4'):
        self.read_path = read_path
        self.write = write
        self.video_type = video_type
        self.model = model
        self.init_param()

    def init_param(self):
        cap = cv2.VideoCapture(self.read_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read_video(self):
        if self.write is not None:
            self.write_video()
        cap = cv2.VideoCapture(self.read_path)
        print("start reading video.")
        num=0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                num+=1
                print(num)
                result = inference_detector(model, frame)
                img=show_result(frame, result, model.CLASSES,ret=True)
                if self.write is not None:
                    self.VideoWriter.write(img)
            else:
                break
        print("end reading video.")

    def write_video(self):
        self.VideoWriter = cv2.VideoWriter(self.write, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, self.size,True)


read_path='/data2/yeliang/data/car_person/b0d1bceaa75259c4a90429c9ce6e21b8.mp4'
write_path="/data2/yeliang/data/car_person/1.mp4"
video_cap=video_io(read_path,write_path)
video_cap.read_video()