from mmcv import Config
import os
from mmdet.apis import init_detector, inference_detector, show_result

config_file = 'configs/pascal_voc/cascade_rcnn_x101_64x4d_fpn_1x_leakage.py'
checkpoint_file = 'work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_leakage/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'data/Leakage/VOC2007/testset/1_Line17_up_20190411032544_74_34km+19.2m_forward.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# show_result(img, result, model.CLASSES,out_file="result.jpg")

# test a list of images and write the results to image files
path = '/data2/yeliang/data/leakage_test'
files= os.listdir(path)
imgs = [os.path.join(path,i) for i in files]
# imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs)):
    show_result(imgs[i], result, model.CLASSES, out_file='results/val_txt/{}'.format(files[i]))











# cfg = Config.fromfile('./configs/cascade_rcnn_r50_fpn_1x.py')
# print(cfg)
# print(cfg.model)
# model=cfg.model
# print(type(model))
# if isinstance(model,list):
#     print("list")
# elif isinstance(model,dict):
#     print("dict")
debug=1