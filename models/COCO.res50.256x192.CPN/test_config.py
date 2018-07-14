import os
import os.path
import sys
import numpy as np

sys.path.insert(0, '../..')
from utils.osutils import add_pypath

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '../..')

    model = 'CPN50'

    num_skeleton = 17
    img_path = os.path.join(root_dir, 'data', 'COCO2017', 'val2017')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15)

    # data augmentation setting
    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (256, 192)
    output_shape = (64, 48)

    # Ground Truth or Detection Result
    use_gt_bbox = True
    if use_gt_bbox:
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_val1.json')
    else:
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'val_dets.json')
    ori_gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'person_keypoints_val2017.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))