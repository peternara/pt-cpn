import sys
import os
import json
from tqdm import tqdm
from utils.osutils import isfile

sys.path.insert(0, 'cocoapi/PythonAPI')
from pycocotools.coco import COCO

annot_root = 'data/COCO2017/annotations'

def anno_transform(gt, det, target):
    target_path = os.path.join(annot_root, target)
    det_path = os.path.join(annot_root, det)
    gt_path = os.path.join(annot_root, gt)


    if isfile(target_path) == True:
        print('The file already exists')
        return
    
    if isfile(det_path) == False:
        print('No original files')
        return
    
    print('The annotation is being transformed')

    eval_gt = COCO(gt_path)
    with open(det_path) as f:
        dets = json.load(f)

    dets = [i for i in dets if i['image_id'] in eval_gt.imgs]
    dets = [i for i in dets if i['category_id'] == 1]
    dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)

    det_anno = []
    
    for anno in tqdm(dets):
        datum = {}
        inp = {}
        img = eval_gt.loadImgs(anno['image_id'])[0]

        x, y, w, h = anno['bbox']
        inp['bbox'] = [int(x), int(y), int(x+w), int(y+h)]

        img_info ={}
        img_info['img_id'] = img['id']
        img_info['img_path'] = img['file_name']

        datum['img_info'] = img_info
        datum['input'] = inp
        datum['score'] = anno['score']
        det_anno.append(datum)
    
    with open(target_path, 'w') as f:
        json.dump(det_anno, f)

    print('done')

anno_transform('person_keypoints_val2017.json','person_detection_minival411_human553.json', 'val_dets.json')    
        