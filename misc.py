import os
import shutil
import cv2

src_path = '/data4/AIData/water/20190703_line17_camera'
dst_path = '/data2/yeliang/data/tunnel_camera/srouce/'


for i in range(1,8):
    src_dir=os.path.join(src_path,'Camera{}'.format(i))
    dst_dir =os.path.join(dst_path,'Camera{}'.format(i))
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for file in os.listdir(src_dir):
        img_path=os.path.join(src_dir,file)
        out_path=os.path.join(dst_dir,file[:-3]+'jpg')
        img=cv2.imread(img_path)
        cv2.imwrite(out_path,img)
