import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse

def inference(args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile('./configs/' + args.cfg_path)

    root='../../dataset/'

    epoch = args.epoch 

    # dataset config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + args.ann_file
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=8
    cfg.gpu_ids = [1]
    cfg.work_dir = args.work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=4,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
    
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    
    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산
    
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds()[i])[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
    submission.head()

    print('inference done!')
    # confusion matrix
    os.system(f'python3 confusion_matrix.py --gt_json {root}{args.ann_file} --pred_csv {os.path.join(args.work_dir_path, args.model_name)}/submission_{epoch}.csv')

if __name__ == "__main__":
    # 파일 기본 위치: baseline/mmdetection/
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=str, default='best_bbox_mAP_50_epoch_30_anno_40')
    parser.add_argument('--ann_file', type=str, default='val_3_cut_40.json')
    parser.add_argument('--work_dir', type=str, default='./work_dirs/faster_rcnn_swin')
    parser.add_argument('--cfg_path', type=str, default='_trash_/faster_rcnn_swin.py')
    parser.add_argument('--model_name', type=str, default='faster_rcnn_swin')
    parser.add_argument('--work_dir_path', type=str, default='./work_dirs')
    args = parser.parse_args()
    inference(args)
