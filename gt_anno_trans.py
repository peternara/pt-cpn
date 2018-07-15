import sys
import os
import json
from tqdm import tqdm
from utils.osutils import isfile

sys.path.insert(0, 'cocoapi/PythonAPI')
from pycocotools.coco import COCO

annot_root = 'data/COCO2017/annotations'

def anno_transform(original, target, is_val):
    target_path = os.path.join(annot_root, target)
    original_path = os.path.join(annot_root, original)

    if isfile(target_path) == True:
        print('The file already exists')
        return
    
    if isfile(original_path) == False:
        print('No original files')
        return
    
    print('The annotation is being transformed')

    ori_anno = COCO(original_path)
    img_ids = ori_anno.getImgIds()
    cat_ids = ori_anno.getCatIds(catNms=['person'])
    train_data = []
    for img_id in tqdm(img_ids):
        img = ori_anno.loadImgs(img_id)[0]
        anno_id = ori_anno.getAnnIds(imgIds=img['id'], catIds=cat_ids)
        anno_files = ori_anno.loadAnns(anno_id)
        
        for anno in anno_files:
            if anno['num_keypoints'] == 0:
                continue
            datum = {}
            inp = {}
            inp['num_keypoints'] = anno['num_keypoints']
            inp['keypoints'] = anno['keypoints']
            x, y, w, h = anno['bbox']
            inp['bbox'] = [int(x), int(y), int(x+w), int(y+h)]

            img_info = {}
            img_info['img_id'] = img_id
            img_info['img_path'] = img['file_name']

            datum['img_info'] = img_info
            datum['input'] = inp

            if is_val == False:
                for i in range(4):
                    temp = datum.copy()
                    temp['operation'] = i
                    train_data.append(temp)
            else:
                datum['score'] = 1
                train_data.append(datum)
        
    with open (target_path, 'w') as f:
        json.dump(train_data, f)
    print('done')

anno_transform('person_keypoints_train2017.json', 'COCO_2017_train1.json', False)
anno_transform('person_keypoints_val2017.json', 'COCO_2017_val1.json', True)    
