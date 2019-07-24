import os
import shutil

# 读取txt文件
# file = '/data2/yeliang/data/tunnel_camera/Camera1_yes.txt'
# img_path = "/data/crop/VOC2007/JPEGImages"
# files=[]
# imgs=[]
# with open(file, 'r') as f:
#     while True:
#         line = f.readline()
#         line = line[:-1]
#         files.append(line+'.jpg')
#         if not line:
#             break
#         src = os.path.join(img_path, line) + ".jpg"
#         imgs.append(src)
# path = '/data2/yeliang/data/leakage_test'
# # files= os.listdir(path)
# imgs = [os.path.join(path,i) for i in files]

def read_txt_cp_file():
    file = '/data2/yeliang/data/tunnel_camera/Camera7_yes.txt'
    img_path = "/data2/yeliang/data/tunnel_camera/source/"
    dst_path = "/data2/yeliang/data/tunnel_camera/source/Camera7_yes"
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    with open(file, 'r') as f:
        while True:
            line = f.readline()
            line = line[:-1]
            if not line:
                break
            line = line.split(' ')[0]
            file = line.split('/')[1]
            src = os.path.join(img_path, line)
            dst = os.path.join(dst_path, file)
            shutil.copy(src, dst)

read_txt_cp_file()