from mmcv import Config

cfg = Config.fromfile('./configs/cascade_rcnn_r50_fpn_1x.py')
print(cfg)
print(cfg.model)
model=cfg.model
print(type(model))
if isinstance(model,list):
    print("list")
elif isinstance(model,dict):
    print("dict")
debug=1