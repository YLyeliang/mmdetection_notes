from mmcv import Config
import os
from mmdet.apis import init_detector, inference_detector, show_result
import datetime

config_file = 'configs/pascal_voc/cascade_rcnn_x101_64x4d_fpn_1x_leakage.py'
# config_file="configs/cascade_rcnn_x101_64x4d_fpn_1x.py" # normal config used to detect coco or voc.
checkpoint_file = 'work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_20190719/epoch_12.pth'  # leakage model.
# checkpoint_file = 'work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_crop/epoch_24.pth'
# checkpoint_file ="work_dirs/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth" # normal pascal vol

# build the model from a config file and a checkpoint file
print("initialize model.")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'data/Leakage/VOC2007/testset/1_Line17_up_20190411032544_74_34km+19.2m_forward.jpg'
# or img = mmcv.imread(img), which will only load it once
# img="people.jpg"
# result = inference_detector(model, img)
# show_result(img, result, model.CLASSES,out_path='./',out_file="people_res.jpg")

# test a list of images and write the results to image files
def read_txt_cp_file(files, imgs):
    file = 'data/crop/VOC2007/ImageSets/Main/test.txt'
    img_path = "data/crop/VOC2007/JPEGImages"
    with open(file, 'r') as f:
        while True:
            line = f.readline()
            line = line[:-1]
            if not line:
                break
            files.append(line + '.jpg')
            src = os.path.join(img_path, line) + ".jpg"
            imgs.append(src)
    return files, imgs


# path = '/data2/yeliang/data/leakage_test'
# files= os.listdir(path)
# imgs = [os.path.join(path,i) for i in files]

# path ='/data2/yeliang/data/Model_comparison/20190719_vs_crop'
# files_= os.listdir(path)
# files=[]
# for i in range(len(files_)):
#     if 'jpg' in files_[i]:
#         files.append(files_[i])
# imgs = [os.path.join(path,i) for i in files]



# files=[]
# imgs=[]
# files,imgs = read_txt_cp_file(files,imgs)

# print("start inference")
# for i, result in enumerate(inference_detector(model, imgs)):
#     show_result(imgs[i], result, model.CLASSES, out_file='/data2/yeliang/data/Model_comparison/20190719_vs_crop/crop_pred/{}'.format(files[i]))


def predict_with_txt(txt_name):
    path = '/data2/yeliang/data/leakage_data/line_10_719/Camera7'
    files = os.listdir(path)
    imgs = [os.path.join(path, i) for i in files]
    print("start inference")
    for i, result in enumerate(inference_detector(model, imgs)):
        show_result(imgs[i], result, model.CLASSES,
                    out_path='/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/Camera7/',
                    out_file='{}'.format(files[i][:-3] + 'jpg'),
                    txt=txt_name)

txt_name = '/data2/yeliang/data/tunnel_camera/20190719_line10_camera/20190719_model/Camera7'
start_time = datetime.datetime.now()
print(start_time)
predict_with_txt(txt_name=txt_name)
end_time = datetime.datetime.now()
print((end_time - start_time).seconds)

