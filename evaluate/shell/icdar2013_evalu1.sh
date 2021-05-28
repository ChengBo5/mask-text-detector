#!/bin/bash

py_dir="tools/test.py"
cfg_dir="evaluate/my_config/2_mask_rcnn/2_mask_rcnn_ICDAR2013.py"
weight_dir="work_dirs/2_mask_rcnn_ICDAR2013/epoch_300.pth" 
log_dir="dataset/result/txt/2_mask_rcnn_ICDAR2013.txt"

fcos_score_thr_list=0.1
fcos_iou_threshold_list=(0.9)
rcnn_score_thr_list=(0.1 0.15 0.2 0.3)
rcnn_iou_threshold_list=(0.2 0.3 0.4 0.5)

for fcos_score_thr in ${fcos_score_thr_list[@]};
do
    for fcos_iou_threshold in ${fcos_iou_threshold_list[@]};
    do
        for rcnn_score_thr in ${rcnn_score_thr_list[@]};
        do
            for rcnn_iou_threshold in ${rcnn_iou_threshold_list[@]};
            do 
                python $py_dir $cfg_dir $weight_dir --show-dir dataset/result/image_result \
                        --cfg-options \
                        model.test_cfg.rpn.nms.iou_threshold=$fcos_iou_threshold \
                        model.test_cfg.rcnn.score_thr=$rcnn_score_thr \
                        model.test_cfg.rcnn.nms.iou_threshold=$rcnn_iou_threshold

                echo "*************rcnn_score_thr="$rcnn_score_thr  $rcnn_iou_threshold  >&1 | tee -a $log_dir ;
                zip -jmq dataset/result/zip/icdar2013_rpn_box.zip dataset/result/icdar2013_rpn_box/* ;
                python evaluate/ICDAR2013/script.py -g=evaluate/ICDAR2013/gt.zip -s=dataset/result/zip/icdar2013_rpn_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;

                zip -jmq dataset/result/zip/icdar2013_mask_box.zip dataset/result/icdar2013_mask_box/* ;
                python evaluate/ICDAR2013/script.py -g=evaluate/ICDAR2013/gt.zip -s=dataset/result/zip/icdar2013_mask_box.zip >&1 | tee -a $log_dir ;
                echo -e >&1 | tee -a $log_dir ;
            done
            echo -e >> $log_dir
            echo -e >> $log_dir
            echo -e >> $log_dir
        done
    done
done
